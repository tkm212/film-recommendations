"""TMDB HTTP client with disk cache for search and movie details."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .constants import (
    CACHE_DIR,
    TMDB_BASE,
    TMDB_CACHE_SLEEP_S,
    TMDB_REQUEST_TIMEOUT_S,
)
from .data_access import create_tmdb_session, get_tmdb_api_key

logger = logging.getLogger(__name__)


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


def cached_get(
    url: str,
    params: dict[str, Any],
    cache_key: str,
    sleep_s: float = TMDB_CACHE_SLEEP_S,
) -> dict[str, Any]:
    """Perform a GET request with disk cache; optional delay to be polite to the API.

    Args:
        url (str): Request URL.
        params (dict[str, Any]): Query parameters for the request.
        cache_key (str): Identifier for the cache file (used with _cache_path).
        sleep_s (float): Seconds to sleep after a live request. Defaults to
            TMDB_CACHE_SLEEP_S.

    Returns:
        dict[str, Any]: Response body as a JSON-decoded dict.

    Raises:
        RuntimeError: If the HTTP request fails (non-2xx status).
    """
    path = _cache_path(cache_key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    session = create_tmdb_session()

    r = session.get(url, params=params, timeout=TMDB_REQUEST_TIMEOUT_S)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        msg = f"TMDB request failed ({r.status_code})."
        raise RuntimeError(msg) from e

    data = r.json()
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    time.sleep(sleep_s)
    return data


def tmdb_search_movie(title: str, year: int | None = None) -> dict[str, Any] | None:
    """Search TMDB for a movie by title and optional year.

    Args:
        title (str): Movie title to search for.
        year (Optional[int]): Optional release year to narrow results.

    Returns:
        Optional[dict[str, Any]]: The best-matching movie result dict from
        TMDB, or None if no results.
    """
    params: dict[str, Any] = {
        "api_key": get_tmdb_api_key(),
        "query": title,
        "include_adult": "false",
    }
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


def tmdb_discover_movies(
    page: int = 1,
    *,
    sort_by: str = "popularity.desc",
    vote_count_gte: int = 50,
    include_adult: bool = False,
) -> dict[str, Any]:
    """Paginated TMDB discover results (broad movie catalog proxy for taste-map background)."""
    params: dict[str, Any] = {
        "api_key": get_tmdb_api_key(),
        "page": page,
        "sort_by": sort_by,
        "vote_count.gte": vote_count_gte,
        "include_adult": str(include_adult).lower(),
    }
    cache_key = f"discover__p{page}__{sort_by}__vc{vote_count_gte}"
    return cached_get(f"{TMDB_BASE}/discover/movie", params=params, cache_key=cache_key)


def tmdb_movie_details(movie_id: int) -> dict[str, Any]:
    """Fetch full TMDB movie details including credits and keywords.

    Args:
        movie_id (int): TMDB movie ID.

    Returns:
        dict[str, Any]: Raw TMDB movie details (genres, countries, overview,
        credits, etc.).
    """
    params = {"api_key": get_tmdb_api_key(), "append_to_response": "credits,keywords"}
    data = cached_get(f"{TMDB_BASE}/movie/{movie_id}", params=params, cache_key=f"movie__{movie_id}__details")
    return data
