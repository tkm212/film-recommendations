"""Shared constants for TMDB access, features, and model training."""

import tempfile
from pathlib import Path

# --- TMDB / HTTP ---
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_API_KEY_ENV_VAR = "API_KEY"
TMDB_DEFAULT_HEADERS = {
    "Accept": "application/json",
}
TMDB_REQUEST_TIMEOUT_S = 30
TMDB_CACHE_SLEEP_S = 0.25
CACHE_DIR = Path(tempfile.mkdtemp(prefix="tmdb_cache_"))

# --- Data paths ---
DEFAULT_RATINGS_PATH = "../inputs/ratings.csv"
DEFAULT_MODEL_PATH = "./letterboxd_tmdb_rating_model.joblib"

# --- Target / split ---
TARGET_COLUMN = "Rating"
DEFAULT_TEST_SIZE = 0.2
N_FOLDS = 3

# --- Sentence embeddings ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_PCA_COMPONENTS = 32
EMBEDDING_CACHE_FILE = "text_embeddings_v2.csv"

# --- Feature columns ---
TEXT_FEATURE_COLUMNS = [
    "overview",
    "keywords",
    "people_text",
    "genres",
    "production_countries",
    "original_language",
]
NUMERIC_FEATURES = [
    "runtime",
    "popularity",
    "vote_average",
    "vote_count",
    "log_budget",
    "log_revenue",
    "log_vote_count",
    "genres_count",
    "countries_count",
    "release_year_tmdb",
]
CATEGORICAL_FEATURES = ["original_language"]
TOP_CAST_COUNT = 5

# --- Model / TF-IDF ---
TFIDF_OVERVIEW_MAX_FEATURES = 2500
TFIDF_KEYWORDS_MAX_FEATURES = 1500
TFIDF_PEOPLE_MAX_FEATURES = 1500
TFIDF_PIPE_MAX_FEATURES = 800
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
RIDGE_ALPHA = 2.0
HGB_SVD_COMPONENTS = 100
RANDOM_STATE = 42
SPARSE_THRESHOLD = 0.3

# --- Predictions / evaluation ---
RATING_CLIP_MIN = 0.5
RATING_CLIP_MAX = 5.0
DEFAULT_SNAP_STEP = 0.5
