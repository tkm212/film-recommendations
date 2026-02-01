"""Tests for film_recoomendations.preprocessing: behaviour and return types."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from film_recoomendations.preprocessing import (
    _cache_path,
    cached_get,
    clean_data,
    create_features,
    extract_features_from_tmdb,
    safe_float,
    safe_int,
    tmdb_movie_details,
    tmdb_search_movie,
)
from conftest import assert_return_type


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
    def test_date_column_type_and_order(
        self, sample_film_ratings_df: pd.DataFrame, check: str
    ) -> None:
        result = clean_data(sample_film_ratings_df)
        if check == "datetime":
            assert pd.api.types.is_datetime64_any_dtype(result["Date"])
        else:
            assert result["Date"].is_monotonic_increasing

    def test_does_not_mutate_input(self, sample_film_ratings_df: pd.DataFrame) -> None:
        original_len = len(sample_film_ratings_df)
        clean_data(sample_film_ratings_df)
        assert len(sample_film_ratings_df) == original_len


class TestCachePath:
    """Behaviour and type tests for _cache_path."""

    @pytest.mark.parametrize(
        "key",
        ["key", "simple_key", "search__Inception (2010)"],
        ids=["plain", "simple_key", "special_chars"],
    )
    def test_returns_path_with_json_suffix(self, key: str) -> None:
        result = _cache_path(key)
        assert_return_type(result, Path)
        assert isinstance(result, Path)
        assert result.suffix == ".json"
        if " " in key or "(" in key:
            assert " " not in result.name or "_" in result.name


class TestCachedGet:
    """Behaviour and type tests for cached_get."""

    def test_returns_dict_type_on_cache_hit(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "cached.json"
        cache_file.write_text(json.dumps({"results": []}), encoding="utf-8")
        with patch("film_recoomendations.preprocessing._cache_path", return_value=cache_file):
            result = cached_get("https://example.com", {}, "key")
        assert_return_type(result, dict)
        assert result == {"results": []}

    def test_returns_dict_on_successful_request(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "cached.json"
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": 1}
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response
        with patch("film_recoomendations.preprocessing._cache_path", return_value=cache_file):
            with patch(
                "film_recoomendations.preprocessing.create_tmdb_session",
                return_value=mock_session,
            ):
                result = cached_get("https://example.com", {"q": "x"}, "key")
        assert result == {"data": 1}
        assert isinstance(result, dict)

    def test_raises_runtime_error_on_http_error(self, tmp_path: Path) -> None:
        import requests as req
        cache_file = tmp_path / "cached.json"
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = req.HTTPError("404")
        mock_session.get.return_value = mock_response
        with patch("film_recoomendations.preprocessing._cache_path", return_value=cache_file):
            with patch(
                "film_recoomendations.preprocessing.create_tmdb_session",
                return_value=mock_session,
            ):
                with pytest.raises(RuntimeError) as exc_info:
                    cached_get("https://example.com", {}, "key")
        assert "TMDB" in str(exc_info.value) or "failed" in str(exc_info.value).lower()


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

    EXPECTED_KEYS = {
        "tmdb_id", "title_tmdb", "release_year_tmdb", "runtime", "budget",
        "revenue", "popularity", "vote_average", "vote_count",
        "original_language", "genres", "production_countries",
        "keywords", "overview", "people_text",
    }

    @pytest.mark.parametrize(
        "details_fixture",
        ["sample_tmdb_details"],
        ids=["full_details"],
    )
    def test_returns_dict_with_expected_keys(
        self, request: pytest.FixtureRequest, details_fixture: str
    ) -> None:
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
        if details is not None:
            details_dict = request.getfixturevalue(details)
        else:
            details_dict = {"id": 1, "title": "X"}
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


class TestTmdbSearchMovie:
    """Behaviour and type tests for tmdb_search_movie."""

    @pytest.mark.parametrize(
        "cached_return,query,expected_none,expected_title",
        [
            ({"results": []}, "Nonexistent Movie 99999", True, None),
            (
                {"results": [{"title": "Inception", "popularity": 10.0}]},
                "Inception",
                False,
                "Inception",
            ),
        ],
        ids=["no_results", "has_results"],
    )
    def test_return_value_from_cached_get(
        self,
        cached_return: dict,
        query: str,
        expected_none: bool,
        expected_title: str | None,
    ) -> None:
        with patch(
            "film_recoomendations.preprocessing.cached_get",
            return_value=cached_return,
        ):
            result = tmdb_search_movie(query)
        if expected_none:
            assert result is None
        else:
            assert result is not None
            assert isinstance(result, dict)
            assert result["title"] == expected_title

    def test_includes_year_in_params_when_given(self) -> None:
        with patch(
            "film_recoomendations.preprocessing.cached_get",
            return_value={"results": [{"title": "Inception", "popularity": 10.0}]},
        ) as mock_get:
            tmdb_search_movie("Inception", year=2010)
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["params"].get("year") == 2010


class TestTmdbMovieDetails:
    """Behaviour and type tests for tmdb_movie_details."""

    def test_returns_dict_type(self) -> None:
        with patch(
            "film_recoomendations.preprocessing.cached_get",
            return_value={"id": 27205, "title": "Inception"},
        ):
            result = tmdb_movie_details(27205)
        assert_return_type(result, dict)
        assert result["id"] == 27205


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
        with patch(
            "film_recoomendations.preprocessing.tmdb_search_movie",
            return_value={"id": 27205},
        ):
            with patch(
                "film_recoomendations.preprocessing.tmdb_movie_details",
                return_value={"id": 27205, "title": "Inception"},
            ):
                with patch(
                    "film_recoomendations.preprocessing.extract_features_from_tmdb",
                    return_value={"tmdb_id": 27205, "title_tmdb": "Inception"},
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
            "film_recoomendations.preprocessing.tmdb_search_movie",
            return_value=None,
        ):
            features_df, missing = create_features(df)
        assert isinstance(missing, list)
        assert len(missing) == 1
        assert missing[0][0] == "Unknown Film XYZ"
        assert len(features_df) == 0
