"""Extract modeling features from TMDB details and enrich ratings DataFrames."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .constants import TOP_CAST_COUNT
from .tmdb_client import tmdb_movie_details, tmdb_search_movie

logger = logging.getLogger(__name__)


def safe_int(x: Any) -> int | float:
    """Convert value to int, or return NaN on failure.

    Args:
        x (Any): Value to convert (e.g. string or number).

    Returns:
        Union[int, float]: int(x) if conversion succeeds, otherwise np.nan.
    """
    try:
        return int(x)
    except Exception:
        return np.nan


def safe_float(x: Any) -> float:
    """Convert value to float, or return NaN on failure.

    Args:
        x (Any): Value to convert (e.g. string or number).

    Returns:
        float: float(x) if conversion succeeds, otherwise np.nan.
    """
    try:
        return float(x)
    except Exception:
        return np.nan


def extract_features_from_tmdb(details: dict[str, Any]) -> dict[str, Any]:
    """Extract a flat feature dict from TMDB movie details for modeling.

    Args:
        details (dict[str, Any]): Raw TMDB movie details (e.g. from
            tmdb_movie_details).

    Returns:
        dict[str, Any]: Flat dict with keys such as tmdb_id, title_tmdb,
        release_year_tmdb, runtime, budget, revenue, popularity,
        vote_average, vote_count, original_language, genres,
        production_countries, keywords, overview, people_text.
    """
    genres = [g.get("name") for g in details.get("genres", []) if g.get("name")]
    countries = [c.get("iso_3166_1") for c in details.get("production_countries", []) if c.get("iso_3166_1")]

    overview = details.get("overview") or ""

    # keywords structure differs sometimes (movie endpoint usually returns {"keywords":[...]} when appended)
    kw_block = details.get("keywords") or {}
    keywords = []
    if isinstance(kw_block, dict):
        keywords = [k.get("name") for k in kw_block.get("keywords", []) if k.get("name")]

    credits_data = details.get("credits") or {}
    crew = credits_data.get("crew", []) if isinstance(credits_data, dict) else []
    cast = credits_data.get("cast", []) if isinstance(credits_data, dict) else []

    directors = [p.get("name") for p in crew if p.get("job") == "Director" and p.get("name")]
    director = directors[0] if directors else ""

    top_cast = [p.get("name") for p in cast[:TOP_CAST_COUNT] if p.get("name")]

    # Combine “people” into a text field so TF-IDF can learn patterns
    people_text = " ".join([director, *top_cast]).strip()

    return {
        "tmdb_id": details.get("id"),
        "title_tmdb": details.get("title"),
        "release_date": details.get("release_date"),
        "release_year_tmdb": safe_int((details.get("release_date") or "")[:4])
        if details.get("release_date")
        else np.nan,
        "runtime": safe_float(details.get("runtime")),
        "budget": safe_float(details.get("budget")),
        "revenue": safe_float(details.get("revenue")),
        "popularity": safe_float(details.get("popularity")),
        "vote_average": safe_float(details.get("vote_average")),
        "vote_count": safe_float(details.get("vote_count")),
        "original_language": details.get("original_language") or "",
        "genres": "|".join([g for g in genres if g]) if genres else "",
        "production_countries": "|".join([c for c in countries if c]) if countries else "",
        "keywords": " ".join([k for k in keywords if k]) if keywords else "",
        "overview": overview,
        "people_text": people_text,
    }


def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int | None]]]:
    """Enrich a ratings DataFrame with TMDB features via search and details APIs.

    Args:
        df (pd.DataFrame): DataFrame with at least columns Name, Year, Date,
            Rating (and optionally Letterboxd URI). Typically the output of
            clean_data.

    Returns:
        tuple[pd.DataFrame, list[tuple[str, Optional[int]]]]: (features_df,
        missing). features_df is a DataFrame with one row per successfully
        enriched film; missing is a list of (title, year) pairs that could
        not be found on TMDB.
    """
    rows = []
    missing = []

    for _, r in tqdm(df.iterrows(), total=len(df)):
        title = str(r["Name"])
        year = int(r["Year"]) if not pd.isna(r["Year"]) else None

        search = tmdb_search_movie(title, year=year)
        if not search:
            missing.append((title, year))
            continue

        movie_id = search.get("id")
        if not movie_id:
            missing.append((title, year))
            continue

        details = tmdb_movie_details(movie_id)
        feats = extract_features_from_tmdb(details)

        out = {
            "Date": r["Date"],
            "Name": title,
            "Year": year,
            "Letterboxd URI": r.get("Letterboxd URI", ""),
            "Rating": float(r["Rating"]),
            **feats,
        }
        rows.append(out)

    features_df = pd.DataFrame(rows)
    logger.info("Missing items count: %d", len(missing))
    logger.debug("Missing items sample (first 20): %s", missing[:20])

    return features_df, missing
