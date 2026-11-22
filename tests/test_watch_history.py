"""Tests for watch history loading."""

from __future__ import annotations

import pandas as pd
import pytest

from film_recommendations.cleaning import clean_watched_data
from film_recommendations.film_universe import merge_catalog_with_personal
from film_recommendations.watch_history import (
    dedupe_watches,
    load_watch_history,
    merge_letterboxd_ratings,
    tag_library_membership,
)


class TestCleanWatched:
    def test_keeps_unrated_rows(self) -> None:
        raw = pd.DataFrame({
            "Date": ["2020-01-01", "2020-02-01"],
            "Name": ["Film A", "Film B"],
            "Year": [2019, 2020],
        })
        out = clean_watched_data(raw)
        assert len(out) == 2
        assert pd.isna(out["Rating"]).all()


class TestWatchHistory:
    def test_diary_plus_ratings_merge(self, tmp_path) -> None:
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        pd.DataFrame({
            "Date": ["2020-01-01", "2020-02-01"],
            "Name": ["Film A", "Film B"],
            "Year": [2019, 2020],
            "Letterboxd URI": ["u1", "u2"],
        }).to_csv(inputs / "diary.csv", index=False)
        pd.DataFrame({
            "Date": ["2020-01-01"],
            "Name": ["Film A"],
            "Year": [2019],
            "Letterboxd URI": ["u1"],
            "Rating": [4.5],
        }).to_csv(inputs / "ratings.csv", index=False)

        watched = dedupe_watches(load_watch_history(tmp_path))
        assert len(watched) == 2
        rated = watched[watched["Name"] == "Film A"]["Rating"].iloc[0]
        assert rated == 4.5
        assert pd.isna(watched[watched["Name"] == "Film B"]["Rating"].iloc[0])

    def test_ratings_only_fallback(self, tmp_path) -> None:
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        pd.DataFrame({
            "Date": ["2020-01-01"],
            "Name": ["Film A"],
            "Year": [2019],
            "Letterboxd URI": ["u1"],
            "Rating": [5],
        }).to_csv(inputs / "ratings.csv", index=False)
        watched = load_watch_history(tmp_path)
        assert len(watched) == 1
        assert watched["Rating"].iloc[0] == 5

    def test_missing_inputs_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_watch_history(tmp_path)


class TestLibraryTagging:
    def test_tag_library_on_catalog_plus_personal(self) -> None:
        catalog = pd.DataFrame({
            "Name": ["In Diary", "Catalog Only"],
            "Year": [2018, 2000],
            "tmdb_id": [1, 2],
        })
        library = pd.DataFrame({
            "Name": ["In Diary"],
            "Year": [2018],
            "Date": pd.to_datetime(["2020-01-01"]),
        })
        universe = merge_catalog_with_personal(catalog, library)
        tagged = tag_library_membership(universe, library)
        assert tagged["in_library"].sum() == 1
        assert bool(tagged.loc[tagged["Name"] == "In Diary", "in_library"].iloc[0])


class TestMergeRatings:
    def test_merge_adds_ratings(self) -> None:
        watched = pd.DataFrame({"Name": ["A"], "Year": [1], "Date": pd.to_datetime(["2020-01-01"])})
        ratings = pd.DataFrame({
            "Name": ["A"],
            "Year": [1],
            "Date": ["2020-01-01"],
            "Rating": [3.5],
        })
        out = merge_letterboxd_ratings(watched, ratings)
        assert out["Rating"].iloc[0] == 3.5
