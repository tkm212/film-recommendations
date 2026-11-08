"""Data access and metric helpers shared by dashboard pages and callbacks."""

import numpy as np
import pandas as pd
import pycountry
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from ..constants import DEFAULT_SNAP_STEP, RATING_CLIP_MAX, RATING_CLIP_MIN

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"

_TEXT_COLUMNS = ("poster_path", "genres", "original_language", "overview", "keywords", "people_text")

_films: pd.DataFrame | None = None
_taste_films: pd.DataFrame | None = None
_importance: dict[str, pd.DataFrame] | None = None
_selected_model: str | None = None
_pca_explained: tuple[float, float, float] | None = None
_taste_embeddings: np.ndarray | None = None


def set_films(films_df: pd.DataFrame) -> None:
    """Register the films-with-predictions DataFrame used by all pages.

    Args:
        films_df (pd.DataFrame): Feature table with Rating, pred, err and
            (optionally) a poster_path column.
    """
    global _films
    df = films_df.copy()

    if "poster_path" not in df.columns:
        df["poster_path"] = ""

    for col in _TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("")

    df["Date"] = pd.to_datetime(df["Date"])
    _films = df


def get_films() -> pd.DataFrame:
    """Return the registered films DataFrame.

    Raises:
        RuntimeError: If set_films has not been called yet.
    """
    if _films is None:
        msg = "Dashboard data not initialised; call set_films() first."
        raise RuntimeError(msg)
    return _films


def set_importance(importance: dict[str, pd.DataFrame] | None) -> None:
    """Register feature-importance tables for the Model analysis page."""
    global _importance
    _importance = importance


def get_importance() -> dict[str, pd.DataFrame]:
    """Return feature-importance tables.

    Raises:
        RuntimeError: If importance has not been registered.
    """
    if _importance is None:
        msg = "Feature importance not available. Retrain with --train to analyse the saved model."
        raise RuntimeError(msg)
    return _importance


def has_importance() -> bool:
    """True when feature-importance tables are registered."""
    return _importance is not None


def set_selected_model(model_name: str | None) -> None:
    """Register the name of the fitted model (shown on Overview)."""
    global _selected_model
    _selected_model = model_name


def get_selected_model() -> str | None:
    """Return the fitted model name, if registered."""
    return _selected_model


def set_pca_explained(explained_variance_ratio: np.ndarray | None) -> None:
    """Register PCA explained-variance ratios for the Taste map axis labels."""
    global _pca_explained
    if explained_variance_ratio is None:
        _pca_explained = None
        return
    padded = np.zeros(3, dtype=np.float64)
    n = min(3, len(explained_variance_ratio))
    padded[:n] = explained_variance_ratio[:n]
    _pca_explained = (float(padded[0]), float(padded[1]), float(padded[2]))


def get_pca_explained() -> tuple[float, float, float] | None:
    """Return PCA variance ratios for PC1-PC3 when the taste map is available."""
    return _pca_explained


def set_taste_films(films_df: pd.DataFrame | None) -> None:
    """Register the taste-map universe (full watched list) and optional library flag."""
    global _taste_films
    if films_df is None:
        _taste_films = None
        return
    df = films_df.copy()
    if "poster_path" not in df.columns:
        df["poster_path"] = ""
    for col in _TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("")
    df["Date"] = pd.to_datetime(df["Date"])
    if "pred" not in df.columns:
        df["pred"] = float("nan")
    if "err" not in df.columns:
        df["err"] = float("nan")
    if "in_library" not in df.columns:
        df["in_library"] = True
    else:
        df["in_library"] = df["in_library"].fillna(False).astype(bool)
    _taste_films = df


def get_taste_films() -> pd.DataFrame:
    """Return the Taste map film table (falls back to rated films if unset)."""
    if _taste_films is not None:
        return _taste_films
    return get_films()


def has_taste_space() -> bool:
    """True when taste map data includes projection coordinates."""
    source = _taste_films if _taste_films is not None else _films
    return source is not None and ("pc1" in source.columns or "um1" in source.columns)


def set_taste_embeddings(embeddings: np.ndarray | None) -> None:
    """Register the full embedding matrix aligned with taste films row order."""
    global _taste_embeddings
    _taste_embeddings = None if embeddings is None else np.asarray(embeddings, dtype=np.float32)


def get_taste_embeddings() -> np.ndarray | None:
    """Return the taste-map embedding matrix when registered."""
    return _taste_embeddings


def library_films() -> pd.DataFrame:
    """Rows flagged as the user's library subset on the taste map."""
    df = get_taste_films()
    if "in_library" in df.columns:
        return df[df["in_library"]].copy()
    return df.copy()


def films_with_clusters(
    n_clusters: int,
    projection: str = "pca",
    subset: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return films with KMeans labels recomputed in the active projection."""
    from ..film_taste_space import cluster_coordinates, projection_columns

    df = (subset if subset is not None else get_taste_films()).copy()
    if not has_taste_space():
        return df
    x_col, y_col, z_col = projection_columns(projection)
    missing = [c for c in (x_col, y_col, z_col) if c not in df.columns]
    if missing:
        return df
    coords = df[[x_col, y_col, z_col]].to_numpy()
    labels = cluster_coordinates(coords, n_clusters)
    df["cluster"] = labels
    df["cluster_label"] = df["cluster"].map(lambda c: f"Cluster {c + 1}")
    return df


def taste_genre_options() -> list[str]:
    """Sorted unique genres across library films on the taste map."""
    genres: set[str] = set()
    for value in library_films()["genres"]:
        genres.update(g for g in str(value).split("|") if g)
    return sorted(genres)


def genre_options() -> list[str]:
    """Sorted unique genres across all films."""
    genres: set[str] = set()
    for value in get_films()["genres"]:
        genres.update(g for g in value.split("|") if g)
    return sorted(genres)


# TMDB uses two codes that are not valid ISO 639-1 and can't be resolved
# from the pycountry database.
_TMDB_LANGUAGE_OVERRIDES = {"cn": "Cantonese", "xx": "No language"}


def language_name(code: str) -> str:
    """Human-readable language name for an ISO 639-1 code (falls back to the code)."""
    if code in _TMDB_LANGUAGE_OVERRIDES:
        return _TMDB_LANGUAGE_OVERRIDES[code]
    language = pycountry.languages.get(alpha_2=code)
    if language is None:
        return code
    return getattr(language, "name", code)


def language_options() -> list[dict]:
    """Dropdown options for languages: full names as labels, codes as values."""
    codes = {lang for lang in get_films()["original_language"] if lang}
    return sorted(
        ({"label": language_name(code), "value": code} for code in codes),
        key=lambda option: option["label"],
    )


def taste_year_bounds() -> tuple[int, int]:
    """(min, max) year watched for library films on the taste map."""
    years = library_films()["Date"].dt.year
    return int(years.min()), int(years.max())


def taste_language_options() -> list[dict]:
    """Language dropdown options for library films on the taste map."""
    codes = {lang for lang in library_films()["original_language"] if lang}
    return sorted(
        ({"label": language_name(code), "value": code} for code in codes),
        key=lambda option: option["label"],
    )


def year_bounds() -> tuple[int, int]:
    """(min, max) year watched across all films."""
    years = get_films()["Date"].dt.year
    return int(years.min()), int(years.max())


def filter_films(
    df: pd.DataFrame,
    genres: list[str] | None = None,
    languages: list[str] | None = None,
    year_range: list[int] | None = None,
    rating_range: list[float] | None = None,
) -> pd.DataFrame:
    """Filter a films DataFrame by genre, language, year watched, and rating.

    Args:
        df (pd.DataFrame): Films DataFrame (genres pipe-separated).
        genres (list[str] | None): Keep films with at least one of these genres.
        languages (list[str] | None): Keep films with one of these languages.
        year_range (list[int] | None): [min, max] year watched (inclusive).
        rating_range (list[float] | None): [min, max] user rating (inclusive).

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    mask = pd.Series(True, index=df.index)

    if genres:
        wanted = set(genres)
        mask &= df["genres"].apply(lambda s: bool(wanted & set(s.split("|"))))
    if languages:
        mask &= df["original_language"].isin(languages)
    if year_range:
        years = df["Date"].dt.year
        mask &= (years >= year_range[0]) & (years <= year_range[1])
    if rating_range:
        at_full_span = rating_range[0] <= RATING_CLIP_MIN and rating_range[1] >= RATING_CLIP_MAX
        if not at_full_span:
            rated = pd.to_numeric(df["Rating"], errors="coerce")
            mask &= rated.notna() & (rated >= rating_range[0]) & (rated <= rating_range[1])

    return df[mask]


def compute_metrics(eval_df: pd.DataFrame) -> dict[str, float]:
    """Compute headline evaluation metrics from a predictions DataFrame.

    Args:
        eval_df (pd.DataFrame): DataFrame with Rating (true) and pred
            (clipped prediction) columns; needs at least 2 rows.

    Returns:
        dict[str, float]: mae, rmse, r2, spearman, exact_match_rate and
        within_half_star.
    """
    y_true = eval_df["Rating"]
    y_pred = eval_df["pred"]
    pred_snapped = np.round(y_pred / DEFAULT_SNAP_STEP) * DEFAULT_SNAP_STEP

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred).correlation,
        "exact_match_rate": (pred_snapped == y_true).mean(),
        "within_half_star": (np.abs(pred_snapped - y_true) <= DEFAULT_SNAP_STEP).mean(),
    }


def poster_url(poster_path: str, size: str = "w92") -> str:
    """Full TMDB image URL for a poster path, or empty string if unavailable.

    Args:
        poster_path (str): TMDB poster path (e.g. "/abc123.jpg").
        size (str): TMDB image size (w92, w185, w342, ...). Defaults to w92.
    """
    if not poster_path or pd.isna(poster_path):
        return ""
    return f"{TMDB_IMAGE_BASE}/{size}{poster_path}"
