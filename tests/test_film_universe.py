"""Tests for TMDB universe catalog helpers."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from film_recommendations.film_universe import (
    build_tmdb_catalog,
    merge_catalog_with_personal,
    personal_letterboxd_union,
)


class TestMergeCatalog:
    def test_personal_only_added_when_missing_from_catalog(self) -> None:
        catalog = pd.DataFrame({
            "Name": ["Blockbuster"],
            "Year": [2020],
            "tmdb_id": [1],
        })
        personal = pd.DataFrame({
            "Name": ["Blockbuster", "Indie Gem"],
            "Year": [2020, 2018],
            "Date": pd.to_datetime(["2020-01-01", "2019-01-01"]),
        })
        merged = merge_catalog_with_personal(catalog, personal)
        assert len(merged) == 2
        assert "Indie Gem" in merged["Name"].values


class TestPersonalUnion:
    def test_watched_and_diary_deduped(self, tmp_path) -> None:
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        pd.DataFrame({
            "Date": ["2020-01-01", "2019-01-01"],
            "Name": ["Shared", "Watched Only"],
            "Year": [2019, 2018],
        }).to_csv(inputs / "watched.csv", index=False)
        pd.DataFrame({
            "Date": ["2020-01-01"],
            "Name": ["Shared"],
            "Year": [2019],
        }).to_csv(inputs / "diary.csv", index=False)

        out = personal_letterboxd_union(tmp_path)
        names = set(out["Name"])
        assert names == {"Shared", "Watched Only"}


class TestBuildCatalog:
    def test_discover_pagination_stops_at_max(self) -> None:
        page1 = {
            "results": [
                {"id": 1, "title": "A", "release_date": "2020-01-01"},
                {"id": 2, "title": "B", "release_date": "2019-01-01"},
            ],
            "total_pages": 10,
        }

        def fake_discover(page: int, **_kwargs) -> dict:
            if page == 1:
                return page1
            return {"results": [], "total_pages": 10}

        with patch("film_recommendations.film_universe.tmdb_discover_movies", side_effect=fake_discover):
            df = build_tmdb_catalog(2, vote_count_gte=0)
        assert len(df) == 2
        assert set(df["tmdb_id"]) == {1, 2}
