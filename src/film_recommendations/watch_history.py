"""Load Letterboxd watch history (watched list, diary, ratings)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cleaning import clean_data, clean_watched_data

DIARY_FILENAME = "diary.csv"
RATINGS_FILENAME = "ratings.csv"
WATCHED_FILENAME = "watched.csv"


def _inputs_dir(repo_root: Path) -> Path:
    return repo_root / "inputs"


def merge_letterboxd_ratings(watched: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Attach star ratings to diary rows where Letterboxd has a rating."""
    rated = ratings.copy()
    if "Rating" not in rated.columns:
        return watched
    rated["Rating"] = pd.to_numeric(rated["Rating"], errors="coerce")
    rated = rated.dropna(subset=["Name"])
    rated = rated.sort_values("Date" if "Date" in rated.columns else "Name").drop_duplicates(
        subset=["Name", "Year"],
        keep="last",
    )
    merge_cols = ["Name", "Year", "Rating"]
    rated_sub = rated[merge_cols]
    base = watched.drop(columns=["Rating"], errors="ignore")
    return base.merge(rated_sub, on=["Name", "Year"], how="left")


def dedupe_watches(watched: pd.DataFrame) -> pd.DataFrame:
    """One row per film (Name + Year), keeping the most recent watch date."""
    return watched.sort_values("Date").drop_duplicates(subset=["Name", "Year"], keep="last").reset_index(drop=True)


def _film_keys(df: pd.DataFrame) -> set[tuple[str, int]]:
    years = pd.to_numeric(df["Year"], errors="coerce")
    keys: set[tuple[str, int]] = set()
    for name, year in zip(df["Name"], years, strict=True):
        if pd.notna(year):
            keys.add((str(name), int(year)))
    return keys


def tag_library_membership(universe: pd.DataFrame, library: pd.DataFrame) -> pd.DataFrame:
    """Mark rows that belong to the user's active library (diary / ratings export)."""
    keys = _film_keys(library)
    out = universe.copy()
    years = pd.to_numeric(out["Year"], errors="coerce")
    out["in_library"] = [
        (str(name), int(year)) in keys if pd.notna(year) else False
        for name, year in zip(out["Name"], years, strict=True)
    ]
    return out


def load_my_films(repo_root: Path) -> pd.DataFrame:
    """Films in diary (+ ratings merge) or ratings-only — your curated watch log."""
    inputs = _inputs_dir(repo_root)
    diary_path = inputs / DIARY_FILENAME
    ratings_path = inputs / RATINGS_FILENAME

    if diary_path.exists():
        watched = clean_watched_data(pd.read_csv(diary_path))
        if ratings_path.exists():
            watched = merge_letterboxd_ratings(watched, pd.read_csv(ratings_path))
        return watched

    if ratings_path.exists():
        return clean_data(pd.read_csv(ratings_path))

    msg = f"No watch history found. Add {inputs / DIARY_FILENAME} or {inputs / RATINGS_FILENAME}."
    raise FileNotFoundError(msg)


def load_film_universe(repo_root: Path) -> pd.DataFrame:
    """All films for the taste-map background (Letterboxd watched.csv when present)."""
    inputs = _inputs_dir(repo_root)
    watched_path = inputs / WATCHED_FILENAME
    ratings_path = inputs / RATINGS_FILENAME

    if watched_path.exists():
        universe = clean_watched_data(pd.read_csv(watched_path))
        if ratings_path.exists():
            universe = merge_letterboxd_ratings(universe, pd.read_csv(ratings_path))
        return dedupe_watches(universe)

    return dedupe_watches(load_my_films(repo_root))


def load_watch_history(repo_root: Path) -> pd.DataFrame:
    """Backward-compatible alias for load_my_films."""
    return load_my_films(repo_root)


def expected_universe_count(repo_root: Path) -> int:
    """Unique films in the current universe export (watched.csv or fallback)."""
    return len(load_film_universe(repo_root))


def expected_watched_count(repo_root: Path) -> int:
    """Number of unique films in the library export."""
    return len(dedupe_watches(load_my_films(repo_root)))
