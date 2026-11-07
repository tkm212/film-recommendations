"""Letterboxd rating model: train and serve a results dashboard.

By default, serves a Dash dashboard from the last saved predictions.

    uv run python scripts/main.py                         # results dashboard
    uv run python scripts/main.py --train                 # benchmark, fit CV-best model, serve
    uv run python scripts/main.py --train --model "Ridge + TF-IDF (baseline)"
    uv run python scripts/main.py --ingest                # TMDB ingest + train + serve

Model comparison dashboard (separate script):

    uv run python scripts/compare.py
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

from film_recommendations.cleaning import clean_data
from film_recommendations.constants import (
    DEFAULT_N_CLUSTERS,
    EMBEDDING_CACHE_FILE,
    N_FOLDS,
    RATED_UMAP_CACHE_FILE,
    TARGET_COLUMN,
    WATCHED_EMBEDDING_CACHE_FILE,
    WATCHED_UMAP_CACHE_FILE,
)
from film_recommendations.dashboard import create_app
from film_recommendations.dashboard import data as dashboard_data
from film_recommendations.data_access import get_film_ratings
from film_recommendations.embeddings import add_merged_text, attach_embeddings, embedding_column_names
from film_recommendations.feature_extraction import create_features
from film_recommendations.feature_importance import analyze_model
from film_recommendations.film_taste_space import attach_taste_coordinates, load_embedding_matrix
from film_recommendations.ml_fitting import (
    create_transformed_features,
    evaluate_rating_predictions,
)
from film_recommendations.model_comparison import (
    build_model_by_name,
    compare_models,
    fit_model,
    resolve_model_name,
)
from film_recommendations.model_specs import MODEL_BUILDERS
from film_recommendations.watch_history import dedupe_watches, load_watch_history

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
RATINGS_FILE = REPO_ROOT / "inputs" / "ratings.csv"
FEATURES_FILE = REPO_ROOT / "outputs" / "ratings_with_tmdb_features.csv"
WATCHED_FEATURES_FILE = REPO_ROOT / "outputs" / "watched_with_tmdb_features.csv"
MODEL_FILE = REPO_ROOT / "models" / "letterboxd_tmdb_rating_model.joblib"
FILMS_FILE = REPO_ROOT / "outputs" / "films_with_predictions.csv"
IMPORTANCE_DIR = REPO_ROOT / "outputs" / "feature_importance"
COMPARISON_FILE = REPO_ROOT / "outputs" / "model_comparison.csv"
SELECTED_MODEL_FILE = REPO_ROOT / "outputs" / "selected_model.txt"
EMBEDDING_CACHE = REPO_ROOT / "outputs" / EMBEDDING_CACHE_FILE
WATCHED_EMBEDDING_CACHE = REPO_ROOT / "outputs" / WATCHED_EMBEDDING_CACHE_FILE
WATCHED_UMAP_CACHE = REPO_ROOT / "outputs" / WATCHED_UMAP_CACHE_FILE
RATED_UMAP_CACHE = REPO_ROOT / "outputs" / RATED_UMAP_CACHE_FILE


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


def ingest_watched_features() -> pd.DataFrame:
    """TMDB-enrich all watched films (diary if present) and cache embeddings for the taste map."""
    watched = dedupe_watches(load_watch_history(REPO_ROOT))
    features_df, missing = create_features(watched)

    WATCHED_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(WATCHED_FEATURES_FILE, index=False)
    attach_embeddings(features_df, WATCHED_EMBEDDING_CACHE)

    logger.info(
        "Enriched %d / %d watched films (%d not on TMDB); cached to %s",
        len(features_df),
        len(watched),
        len(missing),
        WATCHED_FEATURES_FILE,
    )
    return features_df


def ingest_all() -> pd.DataFrame:
    """Ingest rated films for training and full watch history for the taste map."""
    rated = ingest_features()
    ingest_watched_features()
    return rated


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


def _save_importance(importance: dict[str, pd.DataFrame]) -> None:
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in importance.items():
        frame.to_csv(IMPORTANCE_DIR / f"{name}.csv", index=False)
    logger.info("Saved feature importance tables to %s", IMPORTANCE_DIR)


def _save_selected_model(model_name: str) -> None:
    SELECTED_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_MODEL_FILE.write_text(model_name, encoding="utf-8")
    logger.info("Saved selected model name to %s", SELECTED_MODEL_FILE)


def load_selected_model() -> str | None:
    """Load the last fitted model name, if saved."""
    if not SELECTED_MODEL_FILE.exists():
        return None
    return SELECTED_MODEL_FILE.read_text(encoding="utf-8").strip() or None


def analyze_saved_model(X: pd.DataFrame, y: pd.Series) -> dict[str, pd.DataFrame] | None:
    """Load the saved pipeline and compute feature importance when applicable."""
    if not MODEL_FILE.exists():
        logger.warning("Model file not found at %s; skipping feature importance", MODEL_FILE)
        return None
    pipe = joblib.load(MODEL_FILE)
    return analyze_model(pipe, X, y)


def _drop_embedding_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove high-dimensional embedding and helper columns before saving for the dashboard."""
    drop_cols = [col for col in embedding_column_names() if col in df.columns]
    if "merged_text" in df.columns:
        drop_cols.append("merged_text")
    return df.drop(columns=drop_cols)


def _prepare_features(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build model matrix with sentence embeddings and target."""
    features_df, y = create_transformed_features(features_df)
    features_df = attach_embeddings(features_df, EMBEDDING_CACHE)
    features_df = add_merged_text(features_df)
    X = features_df.drop(columns=[TARGET_COLUMN])
    return features_df, y, X


def train(
    features_df: pd.DataFrame,
    model_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame] | None, pd.DataFrame, str]:
    """Benchmark models, fit the chosen pipeline, and save predictions.

    Every film gets an out-of-fold prediction from the selected model. All candidates
    are benchmarked first; model_name defaults to the lowest-MAE CV result.

    Args:
        features_df (pd.DataFrame): TMDB-enriched features DataFrame.
        model_name (str | None): Registered model to fit (--model), or CV best when None.

    Returns:
        tuple: Films with pred/err, optional importance tables, comparison DataFrame,
        and the selected model name.
    """
    features_df, y, X = _prepare_features(features_df)

    comparison, oof_predictions = compare_models(X, y)
    COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_FILE, index=False)
    logger.info("Saved model comparison to %s", COMPARISON_FILE)

    for _, row in comparison.iterrows():
        marker = " ← CV best" if row["is_best"] else ""
        logger.info(
            "  %s: MAE=%.3f, exact=%.1f%%, within 0.5★=%.1f%%%s",
            row["model"],
            row["mae"],
            row["exact_match_rate"] * 100,
            row["within_half_star"] * 100,
            marker,
        )

    selected = resolve_model_name(model_name, comparison)
    logger.info("Fitting model: %s", selected)

    oof_pred = oof_predictions[selected]
    films_df = _drop_embedding_columns(X).assign(Rating=y, pred=oof_pred, err=oof_pred - y)
    results = evaluate_rating_predictions(y, oof_pred, films_df)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    pipe = fit_model(build_model_by_name(selected), X, y, MODEL_FILE)
    logger.info("Saved %s pipeline (fit on all films) to %s", selected, MODEL_FILE)
    _save_selected_model(selected)

    importance = analyze_model(pipe, X, y)
    if importance is not None:
        _save_importance(importance)
    else:
        logger.info("Feature-importance tables skipped (non-linear model selected)")

    FILMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    films_df.to_csv(FILMS_FILE, index=False)
    logger.info("Saved films with OOF predictions to %s", FILMS_FILE)

    logger.info("%d-fold CV MAE (%s): %.3f", N_FOLDS, selected, mean_absolute_error(y, oof_pred))
    logger.info("Exact match rate (nearest 0.5 star): %.1f%%", results["exact_match_rate"] * 100)
    logger.info("Within one snap step (0.5 star):     %.1f%%", results["within_half_star"] * 100)
    return films_df, importance, comparison, selected


def load_importance_for_dashboard() -> dict[str, pd.DataFrame] | None:
    """Load cached importance tables, or recompute from the saved model and features."""
    names = ("coefficients", "groups", "permutation_groups")
    if IMPORTANCE_DIR.exists() and all((IMPORTANCE_DIR / f"{name}.csv").exists() for name in names):
        logger.info("Loading cached feature importance from %s", IMPORTANCE_DIR)
        return {name: pd.read_csv(IMPORTANCE_DIR / f"{name}.csv") for name in names}

    if not FEATURES_FILE.exists():
        logger.warning("Cannot recompute feature importance: %s not found", FEATURES_FILE)
        return None

    logger.info("Recomputing feature importance from saved model and cached features")
    _, y, X = _prepare_features(load_cached_features())
    importance = analyze_saved_model(X, y)
    if importance is not None:
        _save_importance(importance)
    return importance


def attach_taste_map(films_df: pd.DataFrame, embedding_cache: Path) -> pd.DataFrame:
    """Add PCA coordinates and default clusters when embedding cache matches film rows."""
    try:
        embeddings = load_embedding_matrix(embedding_cache)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Skipping taste map: %s", exc)
        dashboard_data.set_pca_explained(None)
        return films_df
    if len(embeddings) != len(films_df):
        logger.warning(
            "Skipping taste map: embedding cache has %d rows, films have %d",
            len(embeddings),
            len(films_df),
        )
        dashboard_data.set_pca_explained(None)
        return films_df
    films_with_taste, explained = attach_taste_coordinates(
        films_df,
        embeddings,
        umap_cache_path=_umap_cache_for(embedding_cache),
        n_clusters=DEFAULT_N_CLUSTERS,
    )
    dashboard_data.set_pca_explained(explained)
    dashboard_data.set_taste_embeddings(embeddings)
    return films_with_taste


def _umap_cache_for(embedding_cache: Path) -> Path:
    if embedding_cache.name == WATCHED_EMBEDDING_CACHE_FILE:
        return WATCHED_UMAP_CACHE
    return RATED_UMAP_CACHE


def _join_predictions(features_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Attach model predictions to watched films where the title was rated."""
    pred_cols = ["Name", "Year", "pred", "err"]
    available = [c for c in pred_cols if c in predictions_df.columns]
    preds = predictions_df[available].drop_duplicates(subset=["Name", "Year"], keep="last")
    merged = features_df.merge(preds, on=["Name", "Year"], how="left")
    if "err" not in merged.columns:
        merged["err"] = float("nan")
    if "pred" not in merged.columns:
        merged["pred"] = float("nan")
    return merged


def prepare_taste_map(predictions_df: pd.DataFrame) -> pd.DataFrame | None:
    """Build taste-map coordinates for all watched films when embeddings are available."""
    expected_n = len(dedupe_watches(load_watch_history(REPO_ROOT)))

    if WATCHED_FEATURES_FILE.exists():
        watched_df = pd.read_csv(WATCHED_FEATURES_FILE, parse_dates=["Date"])
        watched_df = _join_predictions(watched_df, predictions_df)
        cache_path = WATCHED_EMBEDDING_CACHE
    elif FEATURES_FILE.exists():
        rated_features = load_cached_features()
        if expected_n > len(rated_features):
            logger.warning(
                "Watch history has %d unique films but only %d are cached. "
                "Run with --ingest to refresh TMDB features for the taste map.",
                expected_n,
                len(rated_features),
            )
            return None
        watched_df = _join_predictions(rated_features, predictions_df)
        cache_path = EMBEDDING_CACHE
    else:
        logger.warning("No TMDB feature cache found; run with --ingest first.")
        return None

    enriched = attach_taste_map(watched_df, cache_path)
    return enriched if "pc1" in enriched.columns else None


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
        description="Train the rating model and serve a Dash dashboard.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Benchmark all models and fit the selected one before serving.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Re-run TMDB feature engineering (live API calls). Implies --train.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_BUILDERS),
        default=None,
        help="Model to fit when training (default: lowest-MAE from CV benchmark).",
    )
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port (default: 8050).")
    return parser.parse_args()


def main() -> None:
    """Optionally ingest and train, then serve the requested dashboard."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.ingest:
        films_df, importance, _comparison, selected = train(ingest_all(), model_name=args.model)
    elif args.train:
        films_df, importance, _comparison, selected = train(load_cached_features(), model_name=args.model)
    else:
        films_df = load_saved_predictions()
        importance = load_importance_for_dashboard()
        selected = load_selected_model()

    taste_films = prepare_taste_map(films_df)

    app = create_app(
        films_df,
        taste_films=taste_films,
        importance=importance,
        selected_model=selected,
    )
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
