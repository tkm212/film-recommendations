"""Data access and metric helpers shared by dashboard pages and callbacks."""

import numpy as np
import pandas as pd
import pycountry
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from ..constants import DEFAULT_SNAP_STEP

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"

_TEXT_COLUMNS = ("poster_path", "genres", "original_language", "overview", "keywords", "people_text")

_films: pd.DataFrame | None = None


def set_films(films_df: pd.DataFrame) -> None:
    """Register the films-with-predictions DataFrame used by all pages.

    Args:
        films_df (pd.DataFrame): Feature table with Rating, pred, err and
            (optionally) split and poster_path columns.
    """
    global _films
    df = films_df.copy()

    if "poster_path" not in df.columns:
        df["poster_path"] = ""
    if "split" not in df.columns:
        df["split"] = "test"

    for col in _TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("")

    df["Date"] = pd.to_datetime(df["Date"])
    _films = df


def get_films() -> pd.DataFrame:
    """Return the registered films DataFrame.

    Raises:
        RuntimeError: If set_films has not been called yet.
    """
    if _films is None:
        msg = "Dashboard data not initialised; call set_films() first."
        raise RuntimeError(msg)
    return _films


def holdout_films() -> pd.DataFrame:
    """Return only the holdout (test split) films."""
    df = get_films()
    return df[df["split"] == "test"]


def genre_options() -> list[str]:
    """Sorted unique genres across all films."""
    genres: set[str] = set()
    for value in get_films()["genres"]:
        genres.update(g for g in value.split("|") if g)
    return sorted(genres)


# TMDB uses two codes that are not valid ISO 639-1 and can't be resolved
# from the pycountry database.
_TMDB_LANGUAGE_OVERRIDES = {"cn": "Cantonese", "xx": "No language"}


def language_name(code: str) -> str:
    """Human-readable language name for an ISO 639-1 code (falls back to the code)."""
    if code in _TMDB_LANGUAGE_OVERRIDES:
        return _TMDB_LANGUAGE_OVERRIDES[code]
    language = pycountry.languages.get(alpha_2=code)
    if language is None:
        return code
    return getattr(language, "name", code)


def language_options() -> list[dict]:
    """Dropdown options for languages: full names as labels, codes as values."""
    codes = {lang for lang in get_films()["original_language"] if lang}
    return sorted(
        ({"label": language_name(code), "value": code} for code in codes),
        key=lambda option: option["label"],
    )


def year_bounds() -> tuple[int, int]:
    """(min, max) year watched across all films."""
    years = get_films()["Date"].dt.year
    return int(years.min()), int(years.max())


def filter_films(
    df: pd.DataFrame,
    genres: list[str] | None = None,
    languages: list[str] | None = None,
    year_range: list[int] | None = None,
    rating_range: list[float] | None = None,
) -> pd.DataFrame:
    """Filter a films DataFrame by genre, language, year watched, and rating.

    Args:
        df (pd.DataFrame): Films DataFrame (genres pipe-separated).
        genres (list[str] | None): Keep films with at least one of these genres.
        languages (list[str] | None): Keep films with one of these languages.
        year_range (list[int] | None): [min, max] year watched (inclusive).
        rating_range (list[float] | None): [min, max] user rating (inclusive).

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    mask = pd.Series(True, index=df.index)

    if genres:
        wanted = set(genres)
        mask &= df["genres"].apply(lambda s: bool(wanted & set(s.split("|"))))
    if languages:
        mask &= df["original_language"].isin(languages)
    if year_range:
        years = df["Date"].dt.year
        mask &= (years >= year_range[0]) & (years <= year_range[1])
    if rating_range:
        mask &= (df["Rating"] >= rating_range[0]) & (df["Rating"] <= rating_range[1])

    return df[mask]


def compute_metrics(eval_df: pd.DataFrame) -> dict[str, float]:
    """Compute headline evaluation metrics from a predictions DataFrame.

    Args:
        eval_df (pd.DataFrame): DataFrame with Rating (true) and pred
            (clipped prediction) columns; needs at least 2 rows.

    Returns:
        dict[str, float]: mae, rmse, r2, spearman, exact_match_rate and
        within_half_star.
    """
    y_true = eval_df["Rating"]
    y_pred = eval_df["pred"]
    pred_snapped = np.round(y_pred / DEFAULT_SNAP_STEP) * DEFAULT_SNAP_STEP

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred).correlation,
        "exact_match_rate": (pred_snapped == y_true).mean(),
        "within_half_star": (np.abs(pred_snapped - y_true) <= DEFAULT_SNAP_STEP).mean(),
    }


def poster_url(poster_path: str, size: str = "w92") -> str:
    """Full TMDB image URL for a poster path, or empty string if unavailable.

    Args:
        poster_path (str): TMDB poster path (e.g. "/abc123.jpg").
        size (str): TMDB image size (w92, w185, w342, ...). Defaults to w92.
    """
    if not poster_path or pd.isna(poster_path):
        return ""
    return f"{TMDB_IMAGE_BASE}/{size}{poster_path}"
