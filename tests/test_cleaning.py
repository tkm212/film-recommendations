"""Tests for film_recommendations.cleaning: behaviour and return types."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import assert_return_type

from film_recommendations.cleaning import clean_data


class TestCleanData:
    """Behaviour and type tests for clean_data."""

    @pytest.mark.parametrize(
        "fixture_name,expected_len,expected_names,expected_ratings",
        [
            ("sample_film_ratings_df", 3, None, None),
            ("sample_film_ratings_df_with_nulls", 2, {"Film A", "Film C"}, [4.0, 5.0]),
        ],
        ids=["no_nulls", "drops_nulls"],
    )
    def test_returns_dataframe_and_row_counts(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        expected_len: int,
        expected_names: set[str] | None,
        expected_ratings: list[float] | None,
    ) -> None:
        df = request.getfixturevalue(fixture_name)
        result = clean_data(df)
        assert_return_type(result, pd.DataFrame)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == expected_len
        if expected_names is not None:
            assert set(result["Name"]) == expected_names
        if expected_ratings is not None:
            assert list(result["Rating"]) == expected_ratings

    @pytest.mark.parametrize("check", ["datetime", "sorted"], ids=["datetime", "sorted"])
    def test_date_column_type_and_order(self, sample_film_ratings_df: pd.DataFrame, check: str) -> None:
        result = clean_data(sample_film_ratings_df)
        if check == "datetime":
            assert pd.api.types.is_datetime64_any_dtype(result["Date"])
        else:
            assert result["Date"].is_monotonic_increasing

    def test_does_not_mutate_input(self, sample_film_ratings_df: pd.DataFrame) -> None:
        original_len = len(sample_film_ratings_df)
        clean_data(sample_film_ratings_df)
        assert len(sample_film_ratings_df) == original_len
