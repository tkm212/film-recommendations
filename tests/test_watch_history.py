"""Tests for watch history loading."""

from __future__ import annotations

import pandas as pd
import pytest

from film_recommendations.cleaning import clean_watched_data
from film_recommendations.watch_history import (
    dedupe_watches,
    load_film_universe,
    load_my_films,
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


class TestUniverseAndLibrary:
    def test_watched_csv_universe_with_diary_library(self, tmp_path) -> None:
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        pd.DataFrame({
            "Date": ["2020-01-01", "2019-01-01"],
            "Name": ["In Diary", "Only Watched"],
            "Year": [2018, 2017],
        }).to_csv(inputs / "watched.csv", index=False)
        pd.DataFrame({
            "Date": ["2020-01-01"],
            "Name": ["In Diary"],
            "Year": [2018],
        }).to_csv(inputs / "diary.csv", index=False)

        universe = load_film_universe(tmp_path)
        library = load_my_films(tmp_path)
        tagged = tag_library_membership(universe, library)

        assert len(universe) == 2
        assert len(library) == 1
        assert tagged["in_library"].sum() == 1
        assert bool(tagged.loc[tagged["Name"] == "In Diary", "in_library"].iloc[0])

    def test_no_watched_csv_uses_library_as_universe(self, tmp_path) -> None:
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        pd.DataFrame({
            "Date": ["2020-01-01"],
            "Name": ["Film A"],
            "Year": [2019],
            "Rating": [4],
        }).to_csv(inputs / "ratings.csv", index=False)

        universe = load_film_universe(tmp_path)
        library = load_my_films(tmp_path)
        tagged = tag_library_membership(universe, library)

        assert len(universe) == len(library) == 1
        assert tagged["in_library"].all()


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
