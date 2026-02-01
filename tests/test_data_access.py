"""Tests for film_recoomendations.data_access: behaviour and return types."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import requests

from film_recoomendations.data_access import (
    create_tmdb_session,
    get_film_ratings,
    get_tmdb_api_key,
)
from conftest import assert_return_type


class TestGetTmdbApiKey:
    """Behaviour and type tests for get_tmdb_api_key."""

    @pytest.mark.parametrize("env_var,expected", [("API_KEY", "secret123"), ("TMDB_KEY", "custom")])
    def test_returns_str_when_env_set(self, env_var: str, expected: str) -> None:
        with patch.dict("os.environ", {env_var: expected}, clear=False):
            with patch("film_recoomendations.data_access.load_dotenv"):
                result = get_tmdb_api_key(env_var)
        assert result == expected
        assert_return_type(result, str)

    def test_raises_value_error_when_missing(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            with patch("film_recoomendations.data_access.load_dotenv"):
                with patch("os.getenv", return_value=None):
                    with pytest.raises(ValueError) as exc_info:
                        get_tmdb_api_key("API_KEY")
        assert "API_KEY" in str(exc_info.value)
        assert "Missing API key" in str(exc_info.value)


class TestGetFilmRatings:
    """Behaviour and type tests for get_film_ratings."""

    def test_returns_dataframe_with_correct_columns(
        self, ratings_csv_path: Path, sample_film_ratings_df: pd.DataFrame
    ) -> None:
        result = get_film_ratings(ratings_csv_path)
        assert isinstance(result, pd.DataFrame)
        assert_return_type(result, pd.DataFrame)
        assert list(result.columns) == list(sample_film_ratings_df.columns)
        assert len(result) == len(sample_film_ratings_df)

    @pytest.mark.parametrize("path_arg", ["path", "str_path"], ids=["Path", "str"])
    def test_accepts_path_or_str(self, ratings_csv_path: Path, path_arg: str) -> None:
        path = ratings_csv_path if path_arg == "path" else str(ratings_csv_path)
        result = get_film_ratings(path)
        assert isinstance(result, pd.DataFrame)

    def test_raises_file_not_found_for_missing_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError) as exc_info:
            get_film_ratings(missing)
        assert "not found" in str(exc_info.value).lower() or str(missing) in str(exc_info.value)


class TestCreateTmdbSession:
    """Behaviour and type tests for create_tmdb_session."""

    @pytest.mark.parametrize("check", ["return_type", "accept_header"], ids=["type", "header"])
    def test_session_type_and_headers(self, check: str) -> None:
        session = create_tmdb_session()
        if check == "return_type":
            assert_return_type(session, requests.Session)
            assert isinstance(session, requests.Session)
        else:
            assert "Accept" in session.headers
            assert "application/json" in session.headers["Accept"]
