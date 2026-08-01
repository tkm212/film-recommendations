"""Tests for film_recommendations.embeddings."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from film_recommendations.constants import EMBEDDING_DIM
from film_recommendations.embeddings import (
    attach_embeddings,
    combined_film_text,
    embedding_column_names,
    embeddings_dataframe,
)


class TestCombinedFilmText:
    def test_joins_non_empty_fields(self) -> None:
        row = pd.Series({
            "overview": "A plot.",
            "keywords": "war",
            "people_text": "Jane Doe",
            "genres": "Action|Drama",
            "production_countries": "US",
        })
        assert combined_film_text(row) == "A plot. war Jane Doe Action Drama US"

    def test_skips_empty_fields(self) -> None:
        row = pd.Series({
            "overview": "A plot.",
            "keywords": "",
            "people_text": None,
            "genres": "",
            "production_countries": "",
        })
        assert combined_film_text(row) == "A plot."


class TestEmbeddingsDataframe:
    def test_column_names(self) -> None:
        vectors = np.zeros((2, EMBEDDING_DIM), dtype=np.float32)
        df = embeddings_dataframe(vectors)
        assert list(df.columns) == embedding_column_names()
        assert df.shape == (2, EMBEDDING_DIM)


class TestAttachEmbeddings:
    def test_uses_cache_when_row_count_matches(self, tmp_path: Path) -> None:
        features = pd.DataFrame({
            "overview": ["plot"],
            "keywords": ["hero"],
            "people_text": ["actor"],
            "genres": ["Action"],
            "production_countries": ["US"],
        })
        cache = tmp_path / "emb.csv"
        cached = pd.DataFrame({f"emb_{i}": [float(i)] for i in range(EMBEDDING_DIM)})
        cached.to_csv(cache, index=False)

        with patch("film_recommendations.embeddings.compute_text_embeddings") as mock_encode:
            result = attach_embeddings(features, cache)

        mock_encode.assert_not_called()
        assert "emb_0" in result.columns
        assert result["emb_0"].iloc[0] == 0.0

    def test_computes_and_caches_when_missing(self, tmp_path: Path) -> None:
        features = pd.DataFrame({
            "overview": ["plot a", "plot b"],
            "keywords": ["hero", "villain"],
            "people_text": ["actor", "director"],
            "genres": ["Action", "Drama"],
            "production_countries": ["US", "UK"],
        })
        cache = tmp_path / "emb.csv"
        fake_vectors = np.arange(2 * EMBEDDING_DIM, dtype=np.float32).reshape(2, EMBEDDING_DIM)

        with patch(
            "film_recommendations.embeddings.compute_text_embeddings",
            return_value=fake_vectors,
        ) as mock_encode:
            result = attach_embeddings(features, cache)

        mock_encode.assert_called_once()
        assert cache.exists()
        assert result.shape[0] == 2
        assert "emb_383" in result.columns
