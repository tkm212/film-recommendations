import logging
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from .constants import TMDB_DEFAULT_HEADERS

logger = logging.getLogger(__name__)


def get_tmdb_api_key(env_var: str = "API_KEY") -> str:
    """
    Load the TMDB API key from environment variables.

    :param env_var: Name of the environment variable storing the API key
    :return: API key as a string
    :raises ValueError: If the API key is not found
    """
    load_dotenv()
    api_key = os.getenv(env_var)

    if not api_key:
        msg = f"Missing API key. Put {env_var}=... in your .env file."
        raise ValueError(msg)

    return api_key


def get_film_ratings(ratings_path: Path | str = "../inputs/ratings.csv") -> pd.DataFrame:
    """
    Load film ratings data from a CSV file and log basic info.

    :param ratings_path: Path to the ratings CSV
    :param log_head: Whether to log the first few rows
    :return: Pandas DataFrame with ratings data
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
    """
    Create and configure a TMDB requests session.
    """
    session = requests.Session()
    session.headers.update(TMDB_DEFAULT_HEADERS)
    return session
