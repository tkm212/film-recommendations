"""Tests for dashboard filter helpers."""

from __future__ import annotations

import pandas as pd

from film_recommendations.dashboard import data


def test_rating_filter_keeps_unrated_at_full_range() -> None:
    df = pd.DataFrame({
        "Name": ["Rated", "Unrated"],
        "Year": [2020, 2021],
        "Date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        "Rating": [4.0, pd.NA],
        "pred": [3.5, pd.NA],
        "err": [0.5, pd.NA],
        "original_language": ["en", "fr"],
        "genres": ["Drama", "Comedy"],
        "poster_path": ["", ""],
    })
    data.set_taste_films(df)
    out = data.filter_films(df, rating_range=[0.5, 5.0])
    assert len(out) == 2

    narrow = data.filter_films(df, rating_range=[4.0, 5.0])
    assert len(narrow) == 1
    assert narrow.iloc[0]["Name"] == df.iloc[0]["Name"]
