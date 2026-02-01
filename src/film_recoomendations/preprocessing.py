import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

from .constants import CACHE_DIR, TMDB_BASE
from .data_access import create_tmdb_session

logger = logging.getLogger(__name__)

load_dotenv()
TMDB_API_KEY = os.getenv("API_KEY")


def clean_data(film_ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and clean film ratings DataFrame (dates, types, dropna).

    Args:
        film_ratings_df (pd.DataFrame): Raw DataFrame with columns such as
            Date, Name, Year, Rating (and optionally Letterboxd URI).

    Returns:
        pd.DataFrame: A copy with Date as datetime, Year as nullable int,
        Rating as numeric, rows with missing Name or Rating dropped, and
        sorted by Date.
    """
    df = film_ratings_df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    df = df.dropna(subset=["Name", "Rating"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _cache_path(key: str) -> Path:
    """Return the filesystem path for a cache key.

    Args:
        key (str): Arbitrary cache identifier (e.g. "search__title__year").

    Returns:
        Path: Path under CACHE_DIR with key sanitized to a safe filename
        and .json extension.
    """
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", key)
    return CACHE_DIR / f"{safe}.json"


def cached_get(url: str, params: dict[str, Any], cache_key: str, sleep_s: float = 0.25) -> dict[str, Any]:
    """Perform a GET request with disk cache; optional delay to be polite to the API.

    Args:
        url (str): Request URL.
        params (dict[str, Any]): Query parameters for the request.
        cache_key (str): Identifier for the cache file (used with _cache_path).
        sleep_s (float): Seconds to sleep after a live request. Defaults to 0.25.

    Returns:
        dict[str, Any]: Response body as a JSON-decoded dict.

    Raises:
        RuntimeError: If the HTTP request fails (non-2xx status).
    """
    path = _cache_path(cache_key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    session = create_tmdb_session()

    r = session.get(url, params=params, timeout=30)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        msg = f"TMDB request failed ({r.status_code})."
        raise RuntimeError(msg) from e

    data = r.json()
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    time.sleep(sleep_s)
    return data


def tmdb_search_movie(title: str, year: Optional[int] = None) -> Optional[dict[str, Any]]:
    """Search TMDB for a movie by title and optional year.

    Args:
        title (str): Movie title to search for.
        year (Optional[int]): Optional release year to narrow results.

    Returns:
        Optional[dict[str, Any]]: The best-matching movie result dict from
        TMDB, or None if no results.
    """
    params = {"api_key": TMDB_API_KEY, "query": title, "include_adult": "false"}
    if year and not pd.isna(year):
        params["year"] = int(year)

    data = cached_get(f"{TMDB_BASE}/search/movie", params=params, cache_key=f"search__{title}__{year}")

    results = data.get("results", [])
    if not results:
        return None

    # Prefer exact-ish title match if possible, else best popularity
    title_l = title.strip().lower()

    def score(item: dict[str, Any]) -> tuple[int, float]:
        t = (item.get("title") or "").lower()
        ot = (item.get("original_title") or "").lower()
        exact = 2 if (t == title_l or ot == title_l) else (1 if title_l in t else 0)
        pop = item.get("popularity") or 0
        return (exact, pop)

    best = sorted(results, key=score, reverse=True)[0]
    return best


def tmdb_movie_details(movie_id: int) -> dict[str, Any]:
    """Fetch full TMDB movie details including credits and keywords.

    Args:
        movie_id (int): TMDB movie ID.

    Returns:
        dict[str, Any]: Raw TMDB movie details (genres, countries, overview,
        credits, etc.).
    """
    params = {"api_key": TMDB_API_KEY, "append_to_response": "credits,keywords"}
    data = cached_get(f"{TMDB_BASE}/movie/{movie_id}", params=params, cache_key=f"movie__{movie_id}__details")
    return data


def safe_int(x: Any) -> Union[int, float]:
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

    top_cast = [p.get("name") for p in cast[:5] if p.get("name")]

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


def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, Optional[int]]]]:
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
