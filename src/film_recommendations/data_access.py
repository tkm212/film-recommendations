"""Load ratings data and configure TMDB HTTP sessions."""

import logging
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from .constants import DEFAULT_RATINGS_PATH, TMDB_API_KEY_ENV_VAR, TMDB_DEFAULT_HEADERS

logger = logging.getLogger(__name__)


def get_tmdb_api_key(env_var: str = TMDB_API_KEY_ENV_VAR) -> str:
    """Load the TMDB API key from environment variables.

    Args:
        env_var (str): Name of the environment variable storing the API key.
            Defaults to TMDB_API_KEY_ENV_VAR.

    Returns:
        str: The API key.

    Raises:
        ValueError: If the API key is not found in the environment.
    """
    load_dotenv()
    api_key = os.getenv(env_var)

    if not api_key:
        msg = f"Missing API key. Put {env_var}=... in your .env file."
        raise ValueError(msg)

    return api_key


def get_film_ratings(ratings_path: Path | str = DEFAULT_RATINGS_PATH) -> pd.DataFrame:
    """Load film ratings data from a CSV file and log basic info.

    Args:
        ratings_path (Path | str): Path to the ratings CSV file. Defaults to
            DEFAULT_RATINGS_PATH.

    Returns:
        pd.DataFrame: Ratings data (e.g. Date, Name, Year, Rating).

    Raises:
        FileNotFoundError: If the ratings file does not exist.
    """
    ratings_path = Path(ratings_path)

    if not ratings_path.exists():
        msg = f"Ratings file not found: {ratings_path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(ratings_path)

    logger.info("Loaded ratings from %s", ratings_path)
    logger.info("Ratings dataframe shape: %s", df.shape)

    return df


def create_tmdb_session() -> requests.Session:
    """Create and configure a TMDB requests session.

    Returns:
        requests.Session: Session with TMDB default headers
        (Accept: application/json).
    """
    session = requests.Session()
    session.headers.update(TMDB_DEFAULT_HEADERS)
    return session
