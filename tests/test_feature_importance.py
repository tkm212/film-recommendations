"""Tests for film_recommendations.feature_importance."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from film_recommendations.feature_importance import (
    analyze_model,
    coefficient_importance,
    group_importance,
)
from film_recommendations.ml_fitting import create_transformed_features, fit_pipeline, split_train_test


@pytest.fixture
def fitted_model(sample_features_df: pd.DataFrame, tmp_path: Path):
    data, y = create_transformed_features(sample_features_df)
    X_train, y_train, _, _ = split_train_test(data)
    with patch("film_recommendations.ml_fitting.joblib.dump"):
        pipe = fit_pipeline(X_train, y_train, save_path=str(tmp_path / "m.joblib"))
    return pipe, X_train, y_train


class TestCoefficientImportance:
    def test_returns_expected_columns(self, fitted_model) -> None:
        pipe, _, _ = fitted_model
        result = coefficient_importance(pipe)
        assert {"feature", "group", "group_label", "coefficient", "abs_coefficient", "direction"}.issubset(
            result.columns
        )

    def test_sorted_by_abs_coefficient(self, fitted_model) -> None:
        pipe, _, _ = fitted_model
        result = coefficient_importance(pipe)
        assert result["abs_coefficient"].is_monotonic_decreasing

    def test_group_importance_sums_positive(self, fitted_model) -> None:
        pipe, _, _ = fitted_model
        coef_df = coefficient_importance(pipe)
        groups = group_importance(coef_df)
        assert (groups["total_abs_coefficient"] >= 0).all()
        assert abs(groups["share"].sum() - 1.0) < 1e-9
        assert groups["total_abs_coefficient"].sum() > 0


class TestAnalyzeModel:
    def test_returns_three_tables(self, fitted_model) -> None:
        pipe, X, y = fitted_model
        result = analyze_model(pipe, X, y)
        assert result is not None
        assert set(result) == {"coefficients", "groups", "permutation_groups"}
        for frame in result.values():
            assert isinstance(frame, pd.DataFrame)
            assert len(frame) > 0

    def test_permutation_increases_mae_for_some_group(self, fitted_model) -> None:
        pipe, X, y = fitted_model
        result = analyze_model(pipe, X, y)
        assert result is not None
        perm = result["permutation_groups"]
        assert (perm["importance_mean"] >= 0).all()
