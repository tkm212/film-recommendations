"""PCA, UMAP, clustering, and neighbors for films in embedding space."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler

from .constants import DEFAULT_N_CLUSTERS, RANDOM_STATE
from .embeddings import embedding_column_names

PCA_COMPONENTS = 3
UMAP_COLUMNS = ("um1", "um2", "um3")
PCA_COLUMNS = ("pc1", "pc2", "pc3")


@dataclass(frozen=True)
class TasteSpaceFit:
    """PCA coordinates and metadata for a film collection."""

    pc1: np.ndarray
    pc2: np.ndarray
    pc3: np.ndarray
    explained_variance_ratio: np.ndarray
    cluster: np.ndarray


def projection_columns(projection: str) -> tuple[str, str, str]:
    """Axis column names for PCA or UMAP."""
    if projection == "umap":
        return UMAP_COLUMNS
    return PCA_COLUMNS


def _pad_coords_to_3d(coords: np.ndarray) -> np.ndarray:
    """Ensure shape (n_samples, 3) for small or low-rank inputs."""
    n_samples, n_dims = coords.shape
    if n_dims >= PCA_COMPONENTS:
        return coords[:, :PCA_COMPONENTS]
    out = np.zeros((n_samples, PCA_COMPONENTS), dtype=np.float64)
    out[:, :n_dims] = coords
    return out


def fit_pca_coordinates(embeddings: np.ndarray, random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """Standardize embeddings and project to three principal components."""
    n_samples, n_features = embeddings.shape
    n_components = min(PCA_COMPONENTS, n_samples, n_features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(scaled)
    coords_3d = _pad_coords_to_3d(coords)
    explained = np.zeros(PCA_COMPONENTS, dtype=np.float64)
    explained[: len(pca.explained_variance_ratio_)] = pca.explained_variance_ratio_
    return coords_3d, explained


def _has_umap() -> bool:
    return importlib.util.find_spec("umap") is not None


def _fit_nonlinear_coords(scaled: np.ndarray, random_state: int) -> np.ndarray:
    if _has_umap():
        import umap  # type: ignore[import-not-found]

        n_neighbors = max(2, min(15, len(scaled) - 1))
        reducer = umap.UMAP(
            n_components=PCA_COMPONENTS,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
        )
        return _pad_coords_to_3d(reducer.fit_transform(scaled))

    from sklearn.manifold import TSNE

    perplexity = max(2.0, min(30.0, len(scaled) - 1.0))
    tsne = TSNE(
        n_components=PCA_COMPONENTS,
        random_state=random_state,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    return _pad_coords_to_3d(tsne.fit_transform(scaled))


def load_or_fit_umap_coordinates(
    embeddings: np.ndarray,
    cache_path: Path,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Load 3D nonlinear coordinates from cache or fit UMAP (else t-SNE) and save."""
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if len(cached) == len(embeddings) and set(UMAP_COLUMNS).issubset(cached.columns):
            return cached[list(UMAP_COLUMNS)].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    coords = _fit_nonlinear_coords(scaled, random_state)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coords, columns=pd.Index(UMAP_COLUMNS)).to_csv(cache_path, index=False)
    return coords


def cluster_coordinates(coords_3d: np.ndarray, n_clusters: int, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Assign KMeans cluster labels in 3D space."""
    n_films = coords_3d.shape[0]
    if n_films < 2:
        return np.zeros(n_films, dtype=np.int32)
    k = max(2, min(int(n_clusters), n_films))
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return model.fit_predict(coords_3d).astype(np.int32)


def build_taste_space(embeddings: np.ndarray, n_clusters: int = DEFAULT_N_CLUSTERS) -> TasteSpaceFit:
    """PCA to 3D followed by KMeans clustering."""
    coords_3d, explained = fit_pca_coordinates(embeddings)
    labels = cluster_coordinates(coords_3d, n_clusters)
    return TasteSpaceFit(
        pc1=coords_3d[:, 0],
        pc2=coords_3d[:, 1],
        pc3=coords_3d[:, 2],
        explained_variance_ratio=explained,
        cluster=labels,
    )


def attach_taste_coordinates(
    films_df: pd.DataFrame,
    embeddings: np.ndarray,
    umap_cache_path: Path,
    n_clusters: int = DEFAULT_N_CLUSTERS,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Attach PCA, UMAP, and default cluster columns."""
    if len(embeddings) != len(films_df):
        msg = f"Embeddings rows ({len(embeddings)}) must match films ({len(films_df)})."
        raise ValueError(msg)

    pca_coords, explained = fit_pca_coordinates(embeddings)
    umap_coords = load_or_fit_umap_coordinates(embeddings, umap_cache_path)
    labels = cluster_coordinates(pca_coords, n_clusters)

    out = films_df.copy()
    out["pc1"], out["pc2"], out["pc3"] = pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2]
    out["um1"], out["um2"], out["um3"] = umap_coords[:, 0], umap_coords[:, 1], umap_coords[:, 2]
    out["cluster"] = labels
    out["cluster_label"] = out["cluster"].map(lambda c: f"Cluster {c + 1}")
    return out, explained


def apply_taste_space(
    films_df: pd.DataFrame,
    embeddings: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    umap_cache_path: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Backward-compatible wrapper around attach_taste_coordinates."""
    if umap_cache_path is None:
        umap_cache_path = Path("outputs") / "watched_umap_coords.csv"
    return attach_taste_coordinates(films_df, embeddings, umap_cache_path, n_clusters=n_clusters)


def nonlinear_projection_label() -> str:
    """Human-readable label for the nonlinear embedding (UMAP when installed)."""
    if _has_umap():
        return "UMAP (nonlinear)"
    return "t-SNE (nonlinear)"


def load_embedding_matrix(cache_path: Path) -> np.ndarray:
    """Load cached embedding CSV as a float matrix."""
    if not cache_path.exists():
        msg = f"Embedding cache not found: {cache_path}"
        raise FileNotFoundError(msg)
    frame = pd.read_csv(cache_path)
    expected = embedding_column_names()
    if list(frame.columns) == expected:
        return frame.to_numpy(dtype=np.float32)
    if frame.shape[1] == len(expected):
        frame.columns = pd.Index(expected)
        return frame.to_numpy(dtype=np.float32)
    msg = f"Unexpected embedding cache shape: {frame.shape}"
    raise ValueError(msg)


def nearest_neighbor_films(
    films_df: pd.DataFrame,
    embeddings: np.ndarray,
    name: str,
    year: int | float,
    k: int = 8,
) -> pd.DataFrame:
    """Return the k closest films in embedding space (excluding the query)."""
    year_int = int(year)
    names = films_df["Name"].astype(str)
    years = pd.to_numeric(films_df["Year"], errors="coerce")
    match = (names == name) & (years == year_int)
    if not match.any():
        return pd.DataFrame()
    pos = int(match.to_numpy().argmax())
    if pos >= len(embeddings):
        return pd.DataFrame()

    dists = cosine_distances(embeddings[pos : pos + 1], embeddings).ravel()
    order = np.argsort(dists)
    neighbors = [i for i in order if i != pos][:k]
    out = films_df.iloc[neighbors].copy()
    out["similarity"] = 1.0 - dists[neighbors]
    return out.sort_values("similarity", ascending=False)


def _top_tokens(texts: pd.Series, limit: int = 5) -> list[str]:
    from collections import Counter

    counter: Counter[str] = Counter()
    for raw in texts.fillna(""):
        for token in str(raw).split():
            token = token.strip().lower()
            if len(token) > 2:
                counter[token] += 1
    return [word for word, _ in counter.most_common(limit)]


def _top_people(texts: pd.Series, limit: int = 5) -> list[str]:
    """Most common director/cast names in a cluster (pipe-separated people_text)."""
    from collections import Counter

    counter: Counter[str] = Counter()
    for raw in texts.fillna(""):
        s = str(raw).strip()
        if not s:
            continue
        if "|" in s:
            names = [n.strip() for n in s.split("|") if n.strip()]
        else:
            names = [s]
        for name in names:
            counter[name] += 1
    return [name for name, _ in counter.most_common(limit)]


def cluster_insights(films_df: pd.DataFrame) -> list[dict]:
    """Per-cluster summary for label cards: size, genre, keywords, sample films."""
    if "cluster" not in films_df.columns:
        return []

    cards: list[dict] = []
    for cluster_id in sorted(films_df["cluster"].unique()):
        group = films_df[films_df["cluster"] == cluster_id]
        cid = int(cluster_id)
        genres: list[str] = []
        for raw in group["genres"].fillna(""):
            genres.extend(g for g in str(raw).split("|") if g)
        top_genre = pd.Series(genres).mode().iloc[0] if genres else "Mixed"
        keywords = _top_tokens(group["keywords"]) if "keywords" in group.columns else []
        people = _top_people(group["people_text"]) if "people_text" in group.columns else []
        sample = group.head(3)
        cards.append({
            "cluster_label": f"Cluster {cid + 1}",
            "films": len(group),
            "top_genre": top_genre,
            "keywords": keywords,
            "people": people,
            "sample": sample,
        })
    return cards


def cluster_summary(films_df: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster stats: size, mean rating, and most common genre token."""
    if "cluster" not in films_df.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for cluster_id in sorted(films_df["cluster"].unique()):
        group = films_df[films_df["cluster"] == cluster_id]
        cid = int(cluster_id)
        genres: list[str] = []
        for raw in group["genres"].fillna(""):
            genres.extend(g for g in str(raw).split("|") if g)
        top_genre = pd.Series(genres).mode().iloc[0] if genres else "—"
        rows.append({
            "cluster": cid,
            "cluster_label": f"Cluster {cid + 1}",
            "films": len(group),
            "mean_rating": _mean_or_dash(group["Rating"]),
            "mean_pred": _mean_or_dash(group["pred"]) if "pred" in group.columns else "—",
            "top_genre": top_genre,
        })
    return pd.DataFrame(rows).sort_values("cluster")


def _mean_or_dash(series: pd.Series) -> float | str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "—"
    return round(float(values.mean()), 2)
