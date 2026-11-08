"""Tests for film_recommendations.film_taste_space."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from film_recommendations.film_taste_space import (
    attach_taste_coordinates,
    build_taste_space,
    cluster_coordinates,
    cluster_insights,
    cluster_summary,
    fit_pca_coordinates,
    nearest_neighbor_films,
)


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((40, 32)).astype(np.float32)


class TestPcaAndCluster:
    def test_pca_returns_three_columns(self, sample_embeddings: np.ndarray) -> None:
        coords, explained = fit_pca_coordinates(sample_embeddings)
        assert coords.shape == (40, 3)
        assert explained.shape == (3,)
        assert explained.sum() <= 1.0 + 1e-6

    def test_cluster_count_respects_n_films(self, sample_embeddings: np.ndarray) -> None:
        coords, _ = fit_pca_coordinates(sample_embeddings)
        labels = cluster_coordinates(coords, n_clusters=20)
        assert labels.shape == (40,)
        assert len(np.unique(labels)) <= 20

    def test_build_taste_space(self, sample_embeddings: np.ndarray) -> None:
        fit = build_taste_space(sample_embeddings, n_clusters=5)
        assert len(fit.pc1) == 40
        assert len(np.unique(fit.cluster)) <= 5


class TestApplyTasteSpace:
    def test_adds_columns(
        self,
        sample_features_df: pd.DataFrame,
        sample_embeddings: np.ndarray,
        tmp_path,
    ) -> None:
        n = len(sample_features_df)
        emb = sample_embeddings[:n]
        umap_cache = tmp_path / "umap.csv"
        out, _explained = attach_taste_coordinates(
            sample_features_df.assign(Rating=4.0, pred=3.5),
            emb,
            umap_cache,
            n_clusters=4,
        )
        assert {"pc1", "pc2", "pc3", "um1", "um2", "um3", "cluster", "cluster_label"}.issubset(out.columns)

    def test_length_mismatch_raises(self, sample_features_df: pd.DataFrame, tmp_path) -> None:
        with pytest.raises(ValueError, match="must match"):
            attach_taste_coordinates(sample_features_df, np.zeros((2, 32), dtype=np.float32), tmp_path / "u.csv")


class TestNeighborsAndInsights:
    def test_nearest_neighbors(self, sample_features_df: pd.DataFrame, sample_embeddings: np.ndarray, tmp_path) -> None:
        n = len(sample_features_df)
        emb = sample_embeddings[:n]
        films, _ = attach_taste_coordinates(
            sample_features_df.assign(Rating=4.0), emb, tmp_path / "u.csv", n_clusters=3
        )
        query = films.iloc[0]
        nn = nearest_neighbor_films(films, emb, query["Name"], query["Year"], k=3)
        assert len(nn) == 3
        assert "similarity" in nn.columns

    def test_cluster_insights(self, sample_features_df: pd.DataFrame, sample_embeddings: np.ndarray, tmp_path) -> None:
        n = len(sample_features_df)
        emb = sample_embeddings[:n]
        people = ["Christopher Nolan|Leonardo DiCaprio"] * (n // 2) + ["Denis Villeneuve|Timothée Chalamet"] * (
            n - n // 2
        )
        films, _ = attach_taste_coordinates(
            sample_features_df.assign(Rating=4.0, keywords="noir crime", people_text=people),
            emb,
            tmp_path / "u.csv",
            n_clusters=2,
        )
        cards = cluster_insights(films)
        assert cards
        assert "keywords" in cards[0]
        all_people = [p for card in cards for p in card["people"]]
        assert any("Christopher Nolan" in p or "Nolan" in p for p in all_people) or any(
            "Villeneuve" in p for p in all_people
        )


class TestClusterSummary:
    def test_summary_rows(
        self,
        sample_features_df: pd.DataFrame,
        sample_embeddings: np.ndarray,
        tmp_path,
    ) -> None:
        n = len(sample_features_df)
        films = sample_features_df.assign(Rating=4.0, pred=3.5, err=0.0)
        films, _ = attach_taste_coordinates(films, sample_embeddings[:n], tmp_path / "u2.csv", n_clusters=3)
        summary = cluster_summary(films)
        assert len(summary) <= 3
        assert "mean_rating" in summary.columns
