"""TMDB reference catalog for the taste map (Letterboxd has no public film dump)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import clean_data, clean_watched_data
from .constants import DEFAULT_UNIVERSE_MAX_FILMS, UNIVERSE_CATALOG_FILE, UNIVERSE_MIN_VOTE_COUNT
from .tmdb_client import tmdb_discover_movies
from .watch_history import (
    DIARY_FILENAME,
    RATINGS_FILENAME,
    WATCHED_FILENAME,
    dedupe_watches,
    merge_letterboxd_ratings,
)

logger = logging.getLogger(__name__)


def _catalog_path(repo_root: Path) -> Path:
    return repo_root / "outputs" / UNIVERSE_CATALOG_FILE


def _row_from_discover_item(item: dict) -> dict | None:
    title = (item.get("title") or "").strip()
    if not title:
        return None
    movie_id = item.get("id")
    if not movie_id:
        return None
    release = item.get("release_date") or ""
    year = int(release[:4]) if len(release) >= 4 else np.nan
    watched = pd.to_datetime(release, errors="coerce") if release else pd.NaT
    return {
        "Name": title,
        "Year": year,
        "Date": watched,
        "tmdb_id": int(movie_id),
        "Letterboxd URI": "",
        "Rating": np.nan,
    }


def build_tmdb_catalog(
    max_films: int,
    *,
    vote_count_gte: int = UNIVERSE_MIN_VOTE_COUNT,
    catalog_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch up to max_films TMDB movies via discover (popularity-sorted pages)."""
    rows: list[dict] = []
    seen_ids: set[int] = set()
    page = 1

    while len(rows) < max_films:
        data = tmdb_discover_movies(page, vote_count_gte=vote_count_gte)
        results = data.get("results", [])
        if not results:
            break
        for item in results:
            row = _row_from_discover_item(item)
            if row is None or row["tmdb_id"] in seen_ids:
                continue
            seen_ids.add(row["tmdb_id"])
            rows.append(row)
            if len(rows) >= max_films:
                break
        total_pages = int(data.get("total_pages") or page)
        if page >= total_pages:
            break
        page += 1

    df = pd.DataFrame(rows)
    logger.info("Built TMDB universe catalog: %d films (vote_count >= %d)", len(df), vote_count_gte)
    if catalog_path is not None:
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(catalog_path, index=False)
    return df


def load_or_build_tmdb_catalog(
    repo_root: Path,
    max_films: int = DEFAULT_UNIVERSE_MAX_FILMS,
    *,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Load cached TMDB catalog or build it via the API."""
    path = _catalog_path(repo_root)
    if path.exists() and not rebuild:
        return pd.read_csv(path, parse_dates=["Date"])
    return build_tmdb_catalog(max_films, catalog_path=path)


def personal_letterboxd_union(repo_root: Path) -> pd.DataFrame:
    """Union of titles from Letterboxd exports (ensures your films sit on the map)."""
    inputs = repo_root / "inputs"
    frames: list[pd.DataFrame] = []

    ratings_path = inputs / RATINGS_FILENAME
    ratings_df = pd.read_csv(ratings_path) if ratings_path.exists() else None

    for filename in (WATCHED_FILENAME, DIARY_FILENAME):
        path = inputs / filename
        if not path.exists():
            continue
        df = clean_watched_data(pd.read_csv(path))
        if ratings_df is not None and filename != RATINGS_FILENAME:
            df = merge_letterboxd_ratings(df, ratings_df)
        frames.append(df)

    if ratings_df is not None and not any((inputs / f).exists() for f in (WATCHED_FILENAME, DIARY_FILENAME)):
        frames.append(clean_data(ratings_df))

    if not frames:
        return pd.DataFrame(columns=pd.Index(["Name", "Year", "Date", "Rating", "Letterboxd URI"]))

    merged = pd.concat(frames, ignore_index=True)
    return dedupe_watches(merged)


def _catalog_keys(catalog: pd.DataFrame) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    years = pd.to_numeric(catalog["Year"], errors="coerce")
    for name, year in zip(catalog["Name"], years, strict=True):
        if pd.notna(year):
            keys.add((str(name), int(year)))
    return keys


def merge_catalog_with_personal(catalog: pd.DataFrame, personal: pd.DataFrame) -> pd.DataFrame:
    """Append Letterboxd-only titles missing from the TMDB catalog."""
    if personal.empty:
        return catalog.copy()

    cat = catalog.copy()
    if "tmdb_id" not in cat.columns:
        cat["tmdb_id"] = np.nan

    existing = _catalog_keys(cat)
    extras: list[dict] = []
    years = pd.to_numeric(personal["Year"], errors="coerce")
    for _, row in personal.iterrows():
        year = years.loc[row.name]
        if pd.isna(year):
            continue
        key = (str(row["Name"]), int(year))
        if key in existing:
            continue
        extras.append({
            "Name": row["Name"],
            "Year": int(year),
            "Date": row.get("Date", pd.NaT),
            "tmdb_id": np.nan,
            "Letterboxd URI": row.get("Letterboxd URI", ""),
            "Rating": row.get("Rating", np.nan),
        })
        existing.add(key)

    if not extras:
        return cat
    extra_df = pd.DataFrame(extras)
    return pd.concat([cat, extra_df], ignore_index=True)


def load_universe_for_ingest(
    repo_root: Path,
    max_films: int = DEFAULT_UNIVERSE_MAX_FILMS,
    *,
    rebuild_catalog: bool = False,
) -> pd.DataFrame:
    """TMDB catalog plus any Letterboxd titles not already in that catalog."""
    catalog = load_or_build_tmdb_catalog(repo_root, max_films, rebuild=rebuild_catalog)
    personal = personal_letterboxd_union(repo_root)
    return merge_catalog_with_personal(catalog, personal)


def expected_universe_count(
    repo_root: Path,
    max_films: int = DEFAULT_UNIVERSE_MAX_FILMS,
) -> int:
    """Expected rows after catalog merge (cached catalog length + personal extras)."""
    path = _catalog_path(repo_root)
    if not path.exists():
        return max_films
    catalog = pd.read_csv(path)
    personal = personal_letterboxd_union(repo_root)
    if personal.empty:
        return len(catalog)
    merged = merge_catalog_with_personal(catalog, personal)
    return len(merged)
