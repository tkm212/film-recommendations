"""Load Letterboxd watch history (diary + optional ratings merge)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cleaning import clean_data, clean_watched_data

DIARY_FILENAME = "diary.csv"
RATINGS_FILENAME = "ratings.csv"


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


def load_watch_history(repo_root: Path) -> pd.DataFrame:
    """Load all watched films: diary export if present, otherwise ratings export.

    Args:
        repo_root (Path): Repository root (contains inputs/).

    Returns:
        pd.DataFrame: Watched films with Date, Name, Year; Rating when known.

    Raises:
        FileNotFoundError: If neither diary nor ratings CSV exists.
    """
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


def expected_watched_count(repo_root: Path) -> int:
    """Number of unique films in the current watch-history export."""
    return len(dedupe_watches(load_watch_history(repo_root)))
