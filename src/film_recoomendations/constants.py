import tempfile
from pathlib import Path

TMDB_BASE = "https://api.themoviedb.org/3"

CACHE_DIR = Path(tempfile.mkdtemp(prefix="tmdb_cache_"))

TMDB_DEFAULT_HEADERS = {
    "Accept": "application/json",
}