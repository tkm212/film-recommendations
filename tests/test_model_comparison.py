"""Tests for film_recommendations.model_comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from film_recommendations.constants import EMBEDDING_DIM
from film_recommendations.model_comparison import (
    best_model_name,
    compare_models,
    evaluate_predictions,
    resolve_model_name,
)
from film_recommendations.model_specs import MODEL_BUILDERS


@pytest.fixture
def tiny_embedding_features() -> pd.DataFrame:
    """Minimal feature matrix with synthetic embedding columns."""
    n = 12
    rng = np.random.default_rng(42)
    data = {
        "runtime": rng.uniform(80, 140, n),
        "popularity": rng.uniform(1, 10, n),
        "vote_average": rng.uniform(5, 8, n),
        "vote_count": rng.integers(100, 1000, n),
        "log_budget": rng.uniform(10, 15, n),
        "log_revenue": rng.uniform(12, 17, n),
        "log_vote_count": rng.uniform(4, 7, n),
        "genres_count": rng.integers(1, 4, n),
        "countries_count": rng.integers(1, 3, n),
        "release_year_tmdb": [2020.0] * n,
        "original_language": ["en"] * n,
        "overview": ["action drama film"] * n,
        "keywords": ["hero villain"] * n,
        "people_text": ["director actor"] * n,
        "genres": ["Action|Drama"] * n,
        "production_countries": ["US"] * n,
        "merged_text": ["action drama film hero villain director actor"] * n,
    }
    for i in range(EMBEDDING_DIM):
        data[f"emb_{i}"] = rng.normal(size=n)
    return pd.DataFrame(data)


class TestEvaluatePredictions:
    def test_perfect_predictions(self) -> None:
        y = pd.Series([3.0, 4.0, 5.0])
        pred = np.array([3.0, 4.0, 5.0])
        metrics = evaluate_predictions(y, pred)
        assert metrics["mae"] == 0.0
        assert metrics["exact_match_rate"] == 1.0


class TestCompareModels:
    def test_returns_sorted_models(self, tiny_embedding_features: pd.DataFrame) -> None:
        y = pd.Series(np.linspace(3.0, 4.5, len(tiny_embedding_features)))
        result, predictions = compare_models(tiny_embedding_features, y, n_splits=3)
        assert set(result.columns) >= {
            "model",
            "mae",
            "rmse",
            "r2",
            "exact_match_rate",
            "within_half_star",
            "is_best",
            "mae_delta",
        }
        assert len(result) == len(MODEL_BUILDERS)
        assert len(predictions) == len(MODEL_BUILDERS)
        assert result["mae"].is_monotonic_increasing
        assert result["is_best"].sum() == 1
        assert result.iloc[0]["is_best"]


class TestResolveModelName:
    def test_explicit_choice(self, tiny_embedding_features: pd.DataFrame) -> None:
        y = pd.Series(np.linspace(3.0, 4.5, len(tiny_embedding_features)))
        comparison, _ = compare_models(tiny_embedding_features, y, n_splits=3)
        chosen = resolve_model_name("Ridge (metadata only)", comparison)
        assert chosen == "Ridge (metadata only)"

    def test_defaults_to_cv_best(self, tiny_embedding_features: pd.DataFrame) -> None:
        y = pd.Series(np.linspace(3.0, 4.5, len(tiny_embedding_features)))
        comparison, _ = compare_models(tiny_embedding_features, y, n_splits=3)
        assert resolve_model_name(None, comparison) == best_model_name(comparison)

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            resolve_model_name("not a model", pd.DataFrame({"model": ["a"], "mae": [1.0]}))
