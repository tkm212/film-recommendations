"""Shared pytest fixtures and type-checking helpers."""

from __future__ import annotations

import typing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def assert_return_type(
    value: object,
    expected_type: type[typing.Any] | typing.Any,
    *,
    allow_none: bool = False,
) -> None:
    """Assert that a value matches an expected type (for behaviour + type tests).

    Handles common type hints: type, Optional (Union with None), Union, tuple.
    """
    if allow_none and value is None:
        return
    origin = typing.get_origin(expected_type) if hasattr(expected_type, "__origin__") else None
    args = typing.get_args(expected_type) if hasattr(expected_type, "__args__") else ()
    # UnionType (X | Y) in Python 3.10+
    try:
        from types import UnionType

        is_union = origin is typing.Union or origin is UnionType
    except ImportError:
        is_union = origin is typing.Union
    if is_union and args:
        non_none = [a for a in args if a is not type(None)]
        assert isinstance(value, tuple(non_none)), f"expected one of {non_none}, got {type(value)}"
    elif origin is tuple and args:
        assert isinstance(value, tuple), f"expected tuple, got {type(value)}"
        assert len(value) == len(args), f"expected tuple of length {len(args)}, got {len(value)}"
        for v, t in zip(value, args):
            assert_return_type(v, t, allow_none=(t is type(None)))
    elif origin is dict and args:
        key_t, val_t = args
        assert isinstance(value, dict), f"expected dict, got {type(value)}"
        for k, v in value.items():
            assert_return_type(k, key_t, allow_none=False)
            assert_return_type(v, val_t, allow_none=True)
    else:
        assert isinstance(value, expected_type), f"expected {expected_type}, got {type(value)}"


@pytest.fixture
def sample_film_ratings_df() -> pd.DataFrame:
    """Raw film ratings DataFrame as from CSV (mixed types, possible NaNs)."""
    return pd.DataFrame({
        "Date": ["2024-01-15", "2024-02-20", "2024-03-10"],
        "Name": ["Film A", "Film B", "Film C"],
        "Year": [2023, 2022, 2024],
        "Rating": [4.0, 3.5, 5.0],
        "Letterboxd URI": ["https://a", "https://b", "https://c"],
    })


@pytest.fixture
def sample_film_ratings_df_with_nulls() -> pd.DataFrame:
    """Raw ratings with missing Name/Rating to test dropna behaviour."""
    return pd.DataFrame({
        "Date": ["2024-01-15", "2024-02-20", "2024-03-10"],
        "Name": ["Film A", None, "Film C"],
        "Year": [2023, 2022, 2024],
        "Rating": [4.0, np.nan, 5.0],
    })


@pytest.fixture
def sample_tmdb_details() -> dict:
    """Minimal TMDB movie details dict for extract_features_from_tmdb."""
    return {
        "id": 12345,
        "title": "Test Movie",
        "release_date": "2023-06-15",
        "overview": "A test overview.",
        "runtime": 120,
        "budget": 10_000_000,
        "revenue": 50_000_000,
        "popularity": 10.5,
        "vote_average": 7.2,
        "vote_count": 1000,
        "original_language": "en",
        "genres": [{"name": "Drama"}, {"name": "Thriller"}],
        "production_countries": [{"iso_3166_1": "US"}, {"iso_3166_1": "GB"}],
        "keywords": {"keywords": [{"name": "test"}, {"name": "drama"}]},
        "credits": {
            "crew": [{"job": "Director", "name": "Jane Doe"}],
            "cast": [{"name": "Actor One"}, {"name": "Actor Two"}],
        },
    }


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """DataFrame with TMDB-derived columns for create_transformed_features / split_train_test.

    Text columns use repeated terms so TfidfVectorizer(min_df=2) keeps features.
    """
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]),
        "Name": ["A", "B", "C", "D", "E"],
        "Year": [2023] * 5,
        "Rating": [3.0, 3.5, 4.0, 4.5, 5.0],
        "budget": [1e6, 2e6, 3e6, 4e6, 5e6],
        "revenue": [1e7, 2e7, 3e7, 4e7, 5e7],
        "vote_count": [100, 200, 300, 400, 500],
        "genres": ["Drama|Thriller", "Comedy", "Drama", "Action|Comedy", "Drama"],
        "production_countries": ["US", "US|GB", "GB", "US", "US|GB"],
        "overview": ["action drama", "action drama", "comedy", "comedy", "drama"],
        "keywords": ["hero villain", "hero villain", "love", "love", "war"],
        "people_text": ["director a", "director a", "actor b", "actor b", "actor c"],
        "original_language": ["en", "en", "en", "en", "en"],
        "release_year_tmdb": [2023.0] * 5,
        "runtime": [90.0, 100.0, 110.0, 120.0, 130.0],
        "popularity": [1.0, 2.0, 3.0, 4.0, 5.0],
        "vote_average": [6.0, 6.5, 7.0, 7.5, 8.0],
    })


@pytest.fixture
def ratings_csv_path(tmp_path: Path, sample_film_ratings_df: pd.DataFrame) -> Path:
    """Write sample ratings to a CSV and return path."""
    p = tmp_path / "ratings.csv"
    sample_film_ratings_df.to_csv(p, index=False)
    return p
