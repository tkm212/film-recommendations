"""Sentence embeddings for combined film text (overview, keywords, cast/director)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import EMBEDDING_DIM, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

TEXT_SOURCE_COLUMNS = ("overview", "keywords", "people_text", "genres", "production_countries")


def embedding_column_names(dim: int = EMBEDDING_DIM) -> list[str]:
    """Column names for embedding dimensions."""
    return [f"emb_{i}" for i in range(dim)]


def combined_film_text(row: pd.Series) -> str:
    """Merge plot, keywords, people, genres and countries into one embedding document."""
    parts = []
    for col in TEXT_SOURCE_COLUMNS:
        raw = str(row.get(col, "") or "").strip()
        if not raw:
            continue
        if col in {"genres", "production_countries"}:
            raw = raw.replace("|", " ")
        parts.append(raw)
    return " ".join(parts)


def _load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def compute_text_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode documents with a sentence-transformer model."""
    model = _load_embedding_model()
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=len(texts) > 50)
    return np.asarray(vectors, dtype=np.float32)


def embeddings_dataframe(vectors: np.ndarray) -> pd.DataFrame:
    """Wrap embedding matrix as a DataFrame with emb_* column names."""
    columns = embedding_column_names(vectors.shape[1])
    return pd.DataFrame(vectors, columns=pd.Index(columns))


def attach_embeddings(features_df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """Add sentence-embedding columns to a features DataFrame, using a CSV cache."""
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if len(cached) == len(features_df):
            logger.info("Loaded cached text embeddings from %s", cache_path)
            return pd.concat([features_df.reset_index(drop=True), cached.reset_index(drop=True)], axis=1)

    texts = features_df.apply(combined_film_text, axis=1).tolist()
    logger.info("Computing sentence embeddings for %d films (%s)", len(texts), EMBEDDING_MODEL_NAME)
    emb_df = embeddings_dataframe(compute_text_embeddings(texts))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_csv(cache_path, index=False)
    logger.info("Cached text embeddings to %s", cache_path)

    return pd.concat([features_df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)


def add_merged_text(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add a merged_text column used by dense gradient-boosting pipelines."""
    out = features_df.copy()
    out["merged_text"] = out.apply(combined_film_text, axis=1)
    return out
