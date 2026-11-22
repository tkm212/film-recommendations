"""Tests for stochastic discovery recommendations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from film_recommendations.discovery import discover_recommendations


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((40, 32)).astype(np.float32)


class TestDiscoverRecommendations:
    def test_excludes_library_and_varies_with_seed(self, sample_features_df: pd.DataFrame, sample_embeddings) -> None:
        n = len(sample_features_df)
        emb = sample_embeddings[:n]
        universe = sample_features_df.copy()
        library = universe.iloc[:3].copy()
        exclude = {(str(r["Name"]), int(r["Year"])) for _, r in library.iterrows()}

        a = discover_recommendations(universe, emb, library, exclude, n=5, seed=1)
        b = discover_recommendations(universe, emb, library, exclude, n=5, seed=2)
        assert not a.empty
        for _, row in a.iterrows():
            assert (str(row["Name"]), int(row["Year"])) not in exclude

        # Different seeds should usually reorder or change anchors
        assert a["Name"].tolist() != b["Name"].tolist() or not a.equals(b)

    def test_higher_temperature_changes_order(self, sample_features_df: pd.DataFrame, sample_embeddings) -> None:
        n = len(sample_features_df)
        emb = sample_embeddings[:n]
        universe = sample_features_df.copy()
        library = universe.iloc[:2].copy()
        exclude = {(str(r["Name"]), int(r["Year"])) for _, r in library.iterrows()}

        stable = discover_recommendations(universe, emb, library, exclude, n=5, temperature=0.0, seed=99)
        wild = discover_recommendations(universe, emb, library, exclude, n=5, temperature=0.5, seed=99)
        assert len(stable) == len(wild)
        if len(stable) >= 2:
            assert stable["Name"].tolist() != wild["Name"].tolist() or stable.iloc[0]["Name"] != wild.iloc[0]["Name"]
