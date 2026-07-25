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
import plotly.express as px
from dash import Dash, dash_table, dcc, html
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from film_recommendations.cleaning import clean_data
from film_recommendations.constants import DEFAULT_SNAP_STEP, RATING_CLIP_MAX, RATING_CLIP_MIN
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
PREDICTIONS_FILE = REPO_ROOT / "outputs" / "model_predictions.csv"


# --- Pipeline ---


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

    Args:
        features_df (pd.DataFrame): TMDB-enriched features DataFrame.

    Returns:
        pd.DataFrame: Holdout predictions (Date, Name, Year, Rating, pred, err).
    """
    features_df, _ = create_transformed_features(features_df)
    X_train, y_train, X_test, y_test = split_train_test(features_df)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    pipe = fit_pipeline(X_train, y_train, save_path=str(MODEL_FILE))
    logger.info("Saved fitted pipeline to %s", MODEL_FILE)

    eval_df, pred_clipped = make_predictions(pipe, X_test, y_test)
    results = evaluate_rating_predictions(y_test, pred_clipped, eval_df)

    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(PREDICTIONS_FILE, index=False)
    logger.info("Saved holdout predictions to %s", PREDICTIONS_FILE)

    logger.info("Exact match rate (nearest 0.5 star): %.1f%%", results["exact_match_rate"] * 100)
    logger.info("Within one snap step (0.5 star):     %.1f%%", results["within_half_star"] * 100)
    return eval_df


def load_saved_predictions() -> pd.DataFrame:
    """Load the last saved holdout predictions.

    Raises:
        FileNotFoundError: If no predictions have been saved yet.
    """
    if not PREDICTIONS_FILE.exists():
        msg = f"Predictions file not found: {PREDICTIONS_FILE}. Run with --train first."
        raise FileNotFoundError(msg)
    return pd.read_csv(PREDICTIONS_FILE, parse_dates=["Date"])


# --- Dashboard ---

CARD_STYLE = {
    "flex": "1",
    "padding": "16px 20px",
    "borderRadius": "10px",
    "backgroundColor": "#1e2530",
    "color": "#f0f3f7",
    "textAlign": "center",
    "minWidth": "140px",
}
PLOT_TEMPLATE = "plotly_dark"


def compute_metrics(eval_df: pd.DataFrame) -> dict[str, float]:
    """Compute headline evaluation metrics from a predictions DataFrame.

    Args:
        eval_df (pd.DataFrame): DataFrame with Rating (true) and pred
            (clipped prediction) columns.

    Returns:
        dict[str, float]: mae, rmse, r2, spearman, exact_match_rate and
        within_half_star.
    """
    y_true = eval_df["Rating"]
    y_pred = eval_df["pred"]
    pred_snapped = np.round(y_pred / DEFAULT_SNAP_STEP) * DEFAULT_SNAP_STEP

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred).correlation,
        "exact_match_rate": (pred_snapped == y_true).mean(),
        "within_half_star": (np.abs(pred_snapped - y_true) <= DEFAULT_SNAP_STEP).mean(),
    }


def _metric_card(label: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "13px", "color": "#9aa5b1", "marginBottom": "6px"}),
            html.Div(value, style={"fontSize": "26px", "fontWeight": "600"}),
        ],
        style=CARD_STYLE,
    )


def _misses_table(table_id: str, df: pd.DataFrame) -> dash_table.DataTable:
    display = df[["Date", "Name", "Year", "Rating", "pred", "err"]].copy()
    display["Date"] = pd.to_datetime(display["Date"]).dt.date.astype(str)
    display["pred"] = display["pred"].round(2)
    display["err"] = display["err"].round(2)

    return dash_table.DataTable(
        id=table_id,
        data=display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#1e2530", "color": "#f0f3f7", "fontWeight": "600"},
        style_cell={
            "backgroundColor": "#141a22",
            "color": "#dfe5ec",
            "padding": "8px",
            "fontFamily": "system-ui, sans-serif",
            "fontSize": "13px",
            "textAlign": "left",
        },
    )


def build_app(eval_df: pd.DataFrame) -> Dash:
    """Build the Dash app showing model results.

    Args:
        eval_df (pd.DataFrame): Holdout predictions with columns Date, Name,
            Year, Rating, pred, err.

    Returns:
        Dash: The configured Dash application.
    """
    metrics = compute_metrics(eval_df)

    scatter_fig = px.scatter(
        eval_df,
        x="Rating",
        y="pred",
        hover_data=["Name", "Year", "Date"],
        labels={"Rating": "Actual rating", "pred": "Predicted rating"},
        title="Predicted vs actual ratings (holdout set)",
        template=PLOT_TEMPLATE,
    )
    scatter_fig.add_shape(
        type="line",
        x0=RATING_CLIP_MIN,
        y0=RATING_CLIP_MIN,
        x1=RATING_CLIP_MAX,
        y1=RATING_CLIP_MAX,
        line={"dash": "dash", "color": "#9aa5b1"},
    )

    err_fig = px.histogram(
        eval_df,
        x="err",
        nbins=25,
        labels={"err": "Prediction error (pred - actual)"},
        title="Error distribution",
        template=PLOT_TEMPLATE,
    )

    bucket_mae = eval_df.assign(abs_err=lambda d: d["err"].abs()).groupby("Rating")["abs_err"].mean().reset_index()
    bucket_fig = px.bar(
        bucket_mae,
        x="Rating",
        y="abs_err",
        labels={"Rating": "True rating", "abs_err": "Mean absolute error"},
        title="MAE by true rating",
        template=PLOT_TEMPLATE,
    )

    under = eval_df.sort_values("err").head(10)
    over = eval_df.sort_values("err").tail(10).iloc[::-1]

    app = Dash(__name__, title="Letterboxd rating model")
    app.layout = html.Div(
        style={
            "backgroundColor": "#0e1319",
            "minHeight": "100vh",
            "padding": "24px 32px",
            "fontFamily": "system-ui, sans-serif",
        },
        children=[
            html.H1(
                "Letterboxd rating model — Ridge baseline",
                style={"color": "#f0f3f7", "fontSize": "24px", "marginBottom": "4px"},
            ),
            html.P(
                f"Holdout set: {len(eval_df)} most recent ratings",
                style={"color": "#9aa5b1", "marginTop": "0"},
            ),
            html.Div(
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "24px"},
                children=[
                    _metric_card("MAE", f"{metrics['mae']:.3f}"),
                    _metric_card("RMSE", f"{metrics['rmse']:.3f}"),
                    _metric_card("R²", f"{metrics['r2']:.3f}"),
                    _metric_card("Spearman rho", f"{metrics['spearman']:.3f}"),
                    _metric_card("Exact match (0.5★)", f"{metrics['exact_match_rate']:.1%}"),
                    _metric_card("Within 0.5★", f"{metrics['within_half_star']:.1%}"),
                ],
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                children=[
                    dcc.Graph(figure=scatter_fig),
                    dcc.Graph(figure=err_fig),
                ],
            ),
            dcc.Graph(figure=bucket_fig),
            html.H2("Largest under-predictions", style={"color": "#f0f3f7", "fontSize": "18px"}),
            _misses_table("under-table", under),
            html.H2("Largest over-predictions", style={"color": "#f0f3f7", "fontSize": "18px", "marginTop": "24px"}),
            _misses_table("over-table", over),
        ],
    )
    return app


# --- Entry point ---


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
        eval_df = train(ingest_features())
    elif args.train:
        eval_df = train(load_cached_features())
    else:
        eval_df = load_saved_predictions()

    app = build_app(eval_df)
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
