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

import pandas as pd
from sklearn.metrics import mean_absolute_error

from film_recommendations.cleaning import clean_data
from film_recommendations.constants import N_FOLDS, TARGET_COLUMN
from film_recommendations.dashboard import create_app
from film_recommendations.data_access import get_film_ratings
from film_recommendations.feature_extraction import create_features
from film_recommendations.ml_fitting import (
    create_transformed_features,
    cross_validate_predictions,
    evaluate_rating_predictions,
    fit_pipeline,
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
    """Evaluate with k-fold cross-validation and save the model + OOF predictions.

    Every film gets an out-of-fold prediction (from a model that never saw it
    during training), so all films are directly comparable. The saved pipeline
    is then fit on the full dataset for future use.

    Args:
        features_df (pd.DataFrame): TMDB-enriched features DataFrame.

    Returns:
        pd.DataFrame: All films with feature columns plus pred and err columns.
    """
    features_df, y = create_transformed_features(features_df)
    X = features_df.drop(columns=[TARGET_COLUMN])

    oof_pred = cross_validate_predictions(X, y)
    films_df = X.assign(Rating=y, pred=oof_pred, err=oof_pred - y)
    results = evaluate_rating_predictions(y, oof_pred, films_df)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    fit_pipeline(X, y, save_path=str(MODEL_FILE))
    logger.info("Saved pipeline (fit on all films) to %s", MODEL_FILE)

    FILMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    films_df.to_csv(FILMS_FILE, index=False)
    logger.info("Saved films with OOF predictions to %s", FILMS_FILE)

    logger.info("%d-fold CV MAE:                       %.3f", N_FOLDS, mean_absolute_error(y, oof_pred))
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
