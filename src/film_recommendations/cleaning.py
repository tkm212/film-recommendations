"""Clean and normalize Letterboxd ratings DataFrames."""

import pandas as pd


def clean_watched_data(film_ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Letterboxd diary/watch export; rating is optional.

    Args:
        film_ratings_df (pd.DataFrame): Raw export with Date, Name, Year and
            optionally Rating.

    Returns:
        pd.DataFrame: Rows with valid Name, sorted by Date.
    """
    df = film_ratings_df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    else:
        df["Rating"] = pd.NA

    df = df.dropna(subset=["Name"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


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
