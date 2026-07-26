"""Letterboxd rating model: train and serve a results dashboard.

By default, serves a Dash dashboard from the last saved predictions.

    uv run python scripts/main.py            # dashboard from saved predictions
    uv run python scripts/main.py --train    # retrain on cached TMDB features, then serve
    uv run python scripts/main.py --ingest   # re-run TMDB feature engineering (live API
                                             # calls, needs API_KEY), train, then serve
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from film_recommendations.cleaning import clean_data
from film_recommendations.constants import RATING_CLIP_MAX, RATING_CLIP_MIN
from film_recommendations.dashboard import create_app
from film_recommendations.data_access import get_film_ratings
from film_recommendations.feature_extraction import create_features
from film_recommendations.ml_fitting import (
    create_transformed_features,
    evaluate_rating_predictions,
    fit_pipeline,
    make_predictions,
    split_train_test,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
RATINGS_FILE = REPO_ROOT / "inputs" / "ratings.csv"
FEATURES_FILE = REPO_ROOT / "outputs" / "ratings_with_tmdb_features.csv"
MODEL_FILE = REPO_ROOT / "models" / "letterboxd_tmdb_rating_model.joblib"
FILMS_FILE = REPO_ROOT / "outputs" / "films_with_predictions.csv"


def ingest_features() -> pd.DataFrame:
    """Run the TMDB enrichment step from the raw ratings export and cache the result."""
    df = clean_data(get_film_ratings(RATINGS_FILE))
    features_df, missing = create_features(df)

    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(FEATURES_FILE, index=False)

    logger.info(
        "Enriched %d / %d films (%d not found on TMDB); cached features to %s",
        len(features_df),
        len(df),
        len(missing),
        FEATURES_FILE,
    )
    return features_df


def load_cached_features() -> pd.DataFrame:
    """Load the cached TMDB-enriched features CSV.

    Raises:
        FileNotFoundError: If the cached features file does not exist.
    """
    if not FEATURES_FILE.exists():
        msg = f"Cached features file not found: {FEATURES_FILE}. Run with --ingest to generate it."
        raise FileNotFoundError(msg)

    features_df = pd.read_csv(FEATURES_FILE, parse_dates=["Date"])
    logger.info("Loaded %d cached feature rows from %s", len(features_df), FEATURES_FILE)
    return features_df


def train(features_df: pd.DataFrame) -> pd.DataFrame:
    """Fit the model, evaluate on the holdout set, and save model + predictions.

    Predictions are generated for every film (train and test) so the dashboard
    can explore the full collection; evaluation metrics use the test split only.

    Args:
        features_df (pd.DataFrame): TMDB-enriched features DataFrame.

    Returns:
        pd.DataFrame: All films with feature columns plus pred, err and split
        ("train"/"test") columns.
    """
    features_df, _ = create_transformed_features(features_df)
    X_train, y_train, X_test, y_test = split_train_test(features_df)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    pipe = fit_pipeline(X_train, y_train, save_path=str(MODEL_FILE))
    logger.info("Saved fitted pipeline to %s", MODEL_FILE)

    eval_df, pred_clipped = make_predictions(pipe, X_test, y_test)
    results = evaluate_rating_predictions(y_test, pred_clipped, eval_df)

    films_df = pd.concat(
        [
            X_train.assign(Rating=y_train, split="train"),
            X_test.assign(Rating=y_test, split="test"),
        ],
        ignore_index=True,
    )
    films_df["pred"] = np.clip(pipe.predict(films_df), RATING_CLIP_MIN, RATING_CLIP_MAX)
    films_df["err"] = films_df["pred"] - films_df["Rating"]

    FILMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    films_df.to_csv(FILMS_FILE, index=False)
    logger.info("Saved films with predictions to %s", FILMS_FILE)

    logger.info("Exact match rate (nearest 0.5 star): %.1f%%", results["exact_match_rate"] * 100)
    logger.info("Within one snap step (0.5 star):     %.1f%%", results["within_half_star"] * 100)
    return films_df


def load_saved_predictions() -> pd.DataFrame:
    """Load the last saved films-with-predictions table.

    Raises:
        FileNotFoundError: If no predictions have been saved yet.
    """
    if not FILMS_FILE.exists():
        msg = f"Predictions file not found: {FILMS_FILE}. Run with --train first."
        raise FileNotFoundError(msg)
    return pd.read_csv(FILMS_FILE, parse_dates=["Date"])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Serve the Letterboxd rating-model dashboard, optionally retraining first.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Retrain the model on the cached TMDB features before serving the dashboard.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Re-run the heavy TMDB feature engineering (live API calls) from the raw ratings export. Implies --train.",
    )
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port (default: 8050).")
    return parser.parse_args()


def main() -> None:
    """Optionally ingest and train, then serve the results dashboard."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.ingest:
        films_df = train(ingest_features())
    elif args.train:
        films_df = train(load_cached_features())
    else:
        films_df = load_saved_predictions()

    app = create_app(films_df)
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
