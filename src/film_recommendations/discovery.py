"""Stochastic discovery recommendations from embedding space."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import RANDOM_STATE
from .film_taste_space import _film_key, recommend_similar_films


def _pick_anchors(library: pd.DataFrame, n_anchors: int, rng: np.random.Generator) -> pd.DataFrame:
    rated = library.copy()
    if "Rating" in rated.columns:
        stars = pd.to_numeric(rated["Rating"], errors="coerce")
        liked = rated[stars >= 3.5]
        if len(liked) >= n_anchors:
            rated = liked
    n = min(n_anchors, len(rated))
    if n == 0:
        return rated.iloc[0:0]
    idx = rng.choice(len(rated), size=n, replace=False)
    return rated.iloc[idx]


def build_discovery_pool(
    universe: pd.DataFrame,
    embeddings: np.ndarray,
    library: pd.DataFrame,
    exclude_keys: set[tuple[str, int]],
    *,
    n_anchors: int = 6,
    per_anchor: int = 30,
    seed: int | None = None,
) -> pd.DataFrame:
    """Candidate titles similar to random high-rated anchors, excluding the library."""
    rng = np.random.default_rng(seed if seed is not None else RANDOM_STATE)
    anchors = _pick_anchors(library, n_anchors, rng)
    if anchors.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    best_sim: dict[tuple[str, int], float] = {}

    for _, anchor in anchors.iterrows():
        year = anchor["Year"]
        if pd.isna(year):
            continue
        anchor_name = str(anchor["Name"])
        rec = recommend_similar_films(
            universe,
            embeddings,
            anchor_name,
            year,
            exclude_keys,
            k=per_anchor,
            candidate_pool=250,
        )
        if rec.empty:
            continue
        for _, cand in rec.iterrows():
            y = cand["Year"]
            if pd.isna(y):
                continue
            key = _film_key(str(cand["Name"]), int(y))
            sim = float(cand["similarity"])
            if key in best_sim and sim <= best_sim[key]:
                continue
            best_sim[key] = sim
            row = cand.to_dict()
            row["similarity"] = sim
            row["anchor_film"] = anchor_name
            row["anchor_year"] = int(year)
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def sample_discovery_recommendations(
    pool: pd.DataFrame,
    *,
    n: int = 20,
    temperature: float = 0.12,
    seed: int | None = None,
) -> pd.DataFrame:
    """Gumbel-perturbed top picks so refreshes surface different titles."""
    if pool.empty:
        return pool
    rng = np.random.default_rng(seed)
    scores = pool["similarity"].to_numpy(dtype=np.float64)
    uniform = np.clip(rng.uniform(0, 1, size=len(scores)), 1e-9, 1 - 1e-9)
    gumbel = -np.log(-np.log(uniform))
    noisy = scores + temperature * gumbel
    order = np.argsort(-noisy)
    take = min(n, len(order))
    return pool.iloc[order[:take]].reset_index(drop=True)


def discover_recommendations(
    universe: pd.DataFrame,
    embeddings: np.ndarray,
    library: pd.DataFrame,
    exclude_keys: set[tuple[str, int]],
    *,
    n: int = 20,
    temperature: float = 0.12,
    n_anchors: int = 6,
    seed: int | None = None,
) -> pd.DataFrame:
    """End-to-end stochastic discovery list."""
    pool = build_discovery_pool(
        universe,
        embeddings,
        library,
        exclude_keys,
        n_anchors=n_anchors,
        seed=seed,
    )
    return sample_discovery_recommendations(pool, n=n, temperature=temperature, seed=seed)
