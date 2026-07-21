"""Tests for film_recommendations.tmdb_client: behaviour and return types."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from conftest import assert_return_type

from film_recommendations.tmdb_client import (
    _cache_path,
    cached_get,
    tmdb_movie_details,
    tmdb_search_movie,
)


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
        with patch("film_recommendations.tmdb_client._cache_path", return_value=cache_file):
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
        with (
            patch("film_recommendations.tmdb_client._cache_path", return_value=cache_file),
            patch(
                "film_recommendations.tmdb_client.create_tmdb_session",
                return_value=mock_session,
            ),
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
        with (
            patch("film_recommendations.tmdb_client._cache_path", return_value=cache_file),
            patch(
                "film_recommendations.tmdb_client.create_tmdb_session",
                return_value=mock_session,
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            cached_get("https://example.com", {}, "key")
        assert "TMDB" in str(exc_info.value) or "failed" in str(exc_info.value).lower()


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
        with (
            patch("film_recommendations.tmdb_client.get_tmdb_api_key", return_value="test-key"),
            patch(
                "film_recommendations.tmdb_client.cached_get",
                return_value=cached_return,
            ),
        ):
            result = tmdb_search_movie(query)
        if expected_none:
            assert result is None
        else:
            assert result is not None
            assert isinstance(result, dict)
            assert result["title"] == expected_title

    def test_includes_year_in_params_when_given(self) -> None:
        with (
            patch("film_recommendations.tmdb_client.get_tmdb_api_key", return_value="test-key"),
            patch(
                "film_recommendations.tmdb_client.cached_get",
                return_value={"results": [{"title": "Inception", "popularity": 10.0}]},
            ) as mock_get,
        ):
            tmdb_search_movie("Inception", year=2010)
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["params"].get("year") == 2010


class TestTmdbMovieDetails:
    """Behaviour and type tests for tmdb_movie_details."""

    def test_returns_dict_type(self) -> None:
        with (
            patch("film_recommendations.tmdb_client.get_tmdb_api_key", return_value="test-key"),
            patch(
                "film_recommendations.tmdb_client.cached_get",
                return_value={"id": 27205, "title": "Inception"},
            ),
        ):
            result = tmdb_movie_details(27205)
        assert_return_type(result, dict)
        assert result["id"] == 27205
