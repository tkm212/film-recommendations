"""Tests for film_recommendations.feature_extraction: behaviour and return types."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from conftest import assert_return_type

from film_recommendations.feature_extraction import (
    create_features,
    extract_features_from_tmdb,
    safe_float,
    safe_int,
)


class TestSafeInt:
    """Behaviour and type tests for safe_int."""

    @pytest.mark.parametrize("value,expected", [(42, 42), ("42", 42), (42.0, 42)])
    def test_returns_int_for_convertible(self, value: object, expected: int) -> None:
        result = safe_int(value)
        assert result == expected
        assert isinstance(result, (int, float))

    @pytest.mark.parametrize("value", ["x", "not a number", None, []])
    def test_returns_float_nan_for_invalid(self, value: object) -> None:
        result = safe_int(value)
        assert isinstance(result, float)
        assert np.isnan(result)


class TestSafeFloat:
    """Behaviour and type tests for safe_float."""

    @pytest.mark.parametrize("value,expected", [(3.14, 3.14), ("3.14", 3.14), (1, 1.0)])
    def test_returns_float_for_convertible(self, value: object, expected: float) -> None:
        result = safe_float(value)
        assert result == expected
        assert isinstance(result, float)

    @pytest.mark.parametrize("value", ["x", None, []])
    def test_returns_nan_for_invalid(self, value: object) -> None:
        result = safe_float(value)
        assert isinstance(result, float)
        assert np.isnan(result)


class TestExtractFeaturesFromTmdb:
    """Behaviour and type tests for extract_features_from_tmdb."""

    EXPECTED_KEYS: ClassVar[set[str]] = {
        "tmdb_id",
        "title_tmdb",
        "release_year_tmdb",
        "runtime",
        "budget",
        "revenue",
        "popularity",
        "vote_average",
        "vote_count",
        "original_language",
        "genres",
        "production_countries",
        "keywords",
        "overview",
        "people_text",
    }

    @pytest.mark.parametrize(
        "details_fixture",
        ["sample_tmdb_details"],
        ids=["full_details"],
    )
    def test_returns_dict_with_expected_keys(self, request: pytest.FixtureRequest, details_fixture: str) -> None:
        details = request.getfixturevalue(details_fixture)
        result = extract_features_from_tmdb(details)
        assert_return_type(result, dict)
        assert isinstance(result, dict)
        assert self.EXPECTED_KEYS.issubset(result.keys())

    @pytest.mark.parametrize(
        "details,expected_genres,people_contains,overview_empty,runtime_nan",
        [
            (
                "sample_tmdb_details",
                "Drama|Thriller",
                ["Jane Doe", "Actor One"],
                False,
                False,
            ),
            (None, "", [], True, True),
        ],
        ids=["full_details", "minimal"],
    )
    def test_extracted_fields(
        self,
        request: pytest.FixtureRequest,
        details: str | None,
        expected_genres: str,
        people_contains: list[str],
        overview_empty: bool,
        runtime_nan: bool,
    ) -> None:
        details_dict = request.getfixturevalue(details) if details is not None else {"id": 1, "title": "X"}
        result = extract_features_from_tmdb(details_dict)
        if expected_genres:
            assert result["genres"] == expected_genres
        for s in people_contains:
            assert s in result["people_text"]
        if overview_empty:
            assert result["overview"] == ""
        if runtime_nan:
            assert isinstance(result["runtime"], float) and np.isnan(result["runtime"])
        if details is None:
            assert result["tmdb_id"] == 1
            assert result["title_tmdb"] == "X"


class TestCreateFeatures:
    """Behaviour and type tests for create_features."""

    def test_returns_tuple_dataframe_and_list_types(self) -> None:
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01"]),
            "Name": ["Inception"],
            "Year": [2010],
            "Rating": [5.0],
            "Letterboxd URI": [""],
        })
        with (
            patch(
                "film_recommendations.feature_extraction.tmdb_search_movie",
                return_value={"id": 27205},
            ),
            patch(
                "film_recommendations.feature_extraction.tmdb_movie_details",
                return_value={"id": 27205, "title": "Inception"},
            ),
            patch(
                "film_recommendations.feature_extraction.extract_features_from_tmdb",
                return_value={"tmdb_id": 27205, "title_tmdb": "Inception"},
            ),
        ):
            features_df, missing = create_features(df)
        assert isinstance(features_df, pd.DataFrame)
        assert isinstance(missing, list)
        assert len(features_df) == 1
        assert len(missing) == 0

    def test_appends_to_missing_when_search_returns_none(self) -> None:
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01"]),
            "Name": ["Unknown Film XYZ"],
            "Year": [1999],
            "Rating": [3.0],
        })
        with patch(
            "film_recommendations.feature_extraction.tmdb_search_movie",
            return_value=None,
        ):
            features_df, missing = create_features(df)
        assert isinstance(missing, list)
        assert len(missing) == 1
        assert missing[0][0] == "Unknown Film XYZ"
        assert len(features_df) == 0
