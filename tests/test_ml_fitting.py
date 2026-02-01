"""Tests for film_recoomendations.ml_fitting: behaviour and return types."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from film_recoomendations.ml_fitting import (
    create_transformed_features,
    evaluate_rating_predictions,
    fit_pipeline,
    make_predictions,
    split_train_test,
)
from conftest import assert_return_type


class TestCreateTransformedFeatures:
    """Behaviour and type tests for create_transformed_features."""

    def test_returns_tuple_dataframe_and_series_types(
        self, sample_features_df: pd.DataFrame
    ) -> None:
        data, y = create_transformed_features(sample_features_df)
        assert_return_type((data, y), tuple[pd.DataFrame, pd.Series])
        assert isinstance(data, pd.DataFrame)
        assert isinstance(y, pd.Series)

    @pytest.mark.parametrize(
        "col",
        ["log_budget", "log_revenue", "log_vote_count", "genres_count", "countries_count"],
    )
    def test_adds_derived_column(
        self, sample_features_df: pd.DataFrame, col: str
    ) -> None:
        data, y = create_transformed_features(sample_features_df)
        assert col in data.columns

    def test_target_is_rating_series(self, sample_features_df: pd.DataFrame) -> None:
        data, y = create_transformed_features(sample_features_df)
        assert y.name == "Rating" or (y.dtype == float and len(y) == len(sample_features_df))
        pd.testing.assert_series_equal(y, sample_features_df["Rating"].astype(float))

    def test_does_not_mutate_input(self, sample_features_df: pd.DataFrame) -> None:
        original_cols = set(sample_features_df.columns)
        create_transformed_features(sample_features_df)
        assert set(sample_features_df.columns) == original_cols


class TestSplitTrainTest:
    """Behaviour and type tests for split_train_test."""

    def test_returns_four_tuple_types(self, sample_features_df: pd.DataFrame) -> None:
        data, y = create_transformed_features(sample_features_df)
        X_train, y_train, X_test, y_test = split_train_test(data)
        assert_return_type(
            (X_train, y_train, X_test, y_test),
            tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
        )
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)

    def test_train_test_split_by_time(self, sample_features_df: pd.DataFrame) -> None:
        data, _ = create_transformed_features(sample_features_df)
        X_train, y_train, X_test, y_test = split_train_test(data, test_size=0.2)
        n = len(data)
        cut = int(n * 0.8)
        assert len(X_train) == cut
        assert len(X_test) == n - cut
        assert len(y_train) == cut
        assert len(y_test) == n - cut

    def test_rating_not_in_feature_dataframes(self, sample_features_df: pd.DataFrame) -> None:
        data, _ = create_transformed_features(sample_features_df)
        X_train, _, X_test, _ = split_train_test(data)
        assert "Rating" not in X_train.columns
        assert "Rating" not in X_test.columns

    @pytest.mark.parametrize("test_size", [0.1, 0.2, 0.5])
    def test_respects_test_size(
        self, sample_features_df: pd.DataFrame, test_size: float
    ) -> None:
        data, _ = create_transformed_features(sample_features_df)
        _, _, X_test, _ = split_train_test(data, test_size=test_size)
        # Implementation uses last (1 - test_size) for train, rest for test
        train_size = 1 - test_size
        expected_test_len = len(data) - int(len(data) * train_size)
        assert len(X_test) == expected_test_len


class TestFitPipeline:
    """Behaviour and type tests for fit_pipeline."""

    def test_returns_pipeline_type(
        self, sample_features_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        data, y = create_transformed_features(sample_features_df)
        X_train, y_train, _, _ = split_train_test(data)
        save_path = str(tmp_path / "model.joblib")
        with patch("film_recoomendations.ml_fitting.joblib.dump"):
            result = fit_pipeline(X_train, y_train, save_path=save_path)
        assert_return_type(result, Pipeline)
        assert isinstance(result, Pipeline)

    def test_saves_to_disk(
        self, sample_features_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        data, y = create_transformed_features(sample_features_df)
        X_train, y_train, _, _ = split_train_test(data)
        save_path = str(tmp_path / "model.joblib")
        with patch("film_recoomendations.ml_fitting.joblib.dump") as mock_dump:
            fit_pipeline(X_train, y_train, save_path=save_path)
        mock_dump.assert_called_once()
        args, kwargs = mock_dump.call_args
        assert args[1] == save_path

    def test_pipeline_has_preprocess_and_model_steps(
        self, sample_features_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        data, _ = create_transformed_features(sample_features_df)
        X_train, y_train, _, _ = split_train_test(data)
        with patch("film_recoomendations.ml_fitting.joblib.dump"):
            pipe = fit_pipeline(X_train, y_train, save_path=str(tmp_path / "m.joblib"))
        assert "preprocess" in pipe.named_steps
        assert "model" in pipe.named_steps


class TestMakePredictions:
    """Behaviour and type tests for make_predictions."""

    @pytest.fixture
    def fitted_pipe_and_test(
        self, sample_features_df: pd.DataFrame, tmp_path: Path
    ) -> tuple:
        data, _ = create_transformed_features(sample_features_df)
        X_train, y_train, X_test, y_test = split_train_test(data)
        with patch("film_recoomendations.ml_fitting.joblib.dump"):
            pipe = fit_pipeline(X_train, y_train, save_path=str(tmp_path / "m.joblib"))
        return pipe, X_test, y_test

    def test_returns_tuple_dataframe_and_ndarray(
        self, fitted_pipe_and_test: tuple
    ) -> None:
        pipe, X_test, y_test = fitted_pipe_and_test
        eval_df, pred_clipped = make_predictions(pipe, X_test, y_test)
        assert_return_type((eval_df, pred_clipped), tuple[pd.DataFrame, np.ndarray])
        assert isinstance(eval_df, pd.DataFrame)
        assert isinstance(pred_clipped, np.ndarray)

    @pytest.mark.parametrize(
        "col",
        ["Date", "Name", "Year", "Rating", "pred", "err"],
    )
    def test_eval_df_has_column(
        self, fitted_pipe_and_test: tuple, col: str
    ) -> None:
        pipe, X_test, y_test = fitted_pipe_and_test
        eval_df, _ = make_predictions(pipe, X_test, y_test)
        assert col in eval_df.columns

    def test_predictions_clipped_to_half_five(
        self, fitted_pipe_and_test: tuple
    ) -> None:
        pipe, X_test, y_test = fitted_pipe_and_test
        _, pred_clipped = make_predictions(pipe, X_test, y_test)
        assert np.all(pred_clipped >= 0.5)
        assert np.all(pred_clipped <= 5.0)

    def test_err_equals_pred_minus_rating(
        self, fitted_pipe_and_test: tuple
    ) -> None:
        pipe, X_test, y_test = fitted_pipe_and_test
        eval_df, _ = make_predictions(pipe, X_test, y_test)
        np.testing.assert_array_almost_equal(
            eval_df["err"].values,
            eval_df["pred"].values - eval_df["Rating"].values,
        )


EVAL_RESULT_KEYS = ["pred_snapped", "exact_match_rate", "within_half_star", "bucket_mae_by_rating"]


class TestEvaluateRatingPredictions:
    """Behaviour and type tests for evaluate_rating_predictions."""

    @pytest.mark.parametrize(
        "snap_step",
        [0.5, 1.0],
        ids=["snap_0.5", "snap_1.0"],
    )
    def test_returns_dict_with_expected_keys(self, snap_step: float) -> None:
        y_true = pd.Series([4.0, 3.5, 5.0])
        y_pred = np.array([4.0, 3.5, 5.0])
        eval_df = pd.DataFrame({"Rating": [4.0, 3.5, 5.0], "err": [0.0, 0.0, 0.0]})
        result = evaluate_rating_predictions(y_true, y_pred, eval_df, snap_step=snap_step)
        assert_return_type(result, dict)
        assert isinstance(result, dict)
        for key in EVAL_RESULT_KEYS:
            assert key in result

    def test_exact_match_when_pred_equals_true_and_snap_half(self) -> None:
        y_true = pd.Series([4.0, 3.5, 5.0])
        y_pred = np.array([4.0, 3.5, 5.0])
        eval_df = pd.DataFrame({"Rating": [4.0, 3.5, 5.0], "err": [0.0, 0.0, 0.0]})
        result = evaluate_rating_predictions(y_true, y_pred, eval_df, snap_step=0.5)
        assert result["exact_match_rate"] == 1.0
        assert result["within_half_star"] == 1.0

    @pytest.mark.parametrize(
        "key",
        EVAL_RESULT_KEYS,
    )
    def test_result_has_key(self, key: str) -> None:
        y_true = pd.Series([4.0, 3.5])
        y_pred = np.array([4.0, 3.5])
        eval_df = pd.DataFrame({"Rating": [4.0, 3.5], "err": [0.0, 0.0]})
        result = evaluate_rating_predictions(y_true, y_pred, eval_df)
        assert key in result

    def test_bucket_mae_by_rating_is_series(self) -> None:
        y_true = pd.Series([4.0, 4.0, 5.0])
        y_pred = np.array([4.0, 4.0, 5.0])
        eval_df = pd.DataFrame({"Rating": [4.0, 4.0, 5.0], "err": [0.0, 0.0, 0.0]})
        result = evaluate_rating_predictions(y_true, y_pred, eval_df)
        assert isinstance(result["bucket_mae_by_rating"], pd.Series)
