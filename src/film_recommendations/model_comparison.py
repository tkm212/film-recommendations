"""Cross-validated comparison of rating-model variants."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .constants import DEFAULT_SNAP_STEP, N_FOLDS, RANDOM_STATE, RATING_CLIP_MAX, RATING_CLIP_MIN
from .model_specs import MODEL_BUILDERS

logger = logging.getLogger(__name__)


def _snap_metrics(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float]:
    pred_snapped = np.round(y_pred / DEFAULT_SNAP_STEP) * DEFAULT_SNAP_STEP
    exact = (pred_snapped == y_true).mean()
    within = (np.abs(pred_snapped - y_true) <= DEFAULT_SNAP_STEP).mean()
    return float(exact), float(within)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute headline metrics for clipped out-of-fold predictions."""
    clipped = np.clip(y_pred, RATING_CLIP_MIN, RATING_CLIP_MAX)
    exact, within = _snap_metrics(y_true, clipped)
    return {
        "mae": mean_absolute_error(y_true, clipped),
        "rmse": root_mean_squared_error(y_true, clipped),
        "r2": r2_score(y_true, clipped),
        "exact_match_rate": exact,
        "within_half_star": within,
    }


def cross_validate_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = N_FOLDS) -> np.ndarray:
    """Out-of-fold predictions for one pipeline."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(pipe, X, y, cv=cv)
    return np.clip(pred, RATING_CLIP_MIN, RATING_CLIP_MAX)


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_FOLDS,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Benchmark all registered models with the same k-fold CV protocol.

    Args:
        X (pd.DataFrame): Feature matrix (must include emb_* columns for embedding models).
        y (pd.Series): Target ratings.
        n_splits (int): Number of CV folds. Defaults to N_FOLDS.

    Returns:
        tuple[pd.DataFrame, dict[str, np.ndarray]]: Metrics table (sorted by MAE, with
        is_best and mae_delta) and out-of-fold predictions keyed by model name.
    """
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for name, builder in MODEL_BUILDERS.items():
        logger.info("Cross-validating model: %s", name)
        try:
            pipe = builder()
            pred = cross_validate_model(pipe, X, y, n_splits=n_splits)
            predictions[name] = pred
            metrics = evaluate_predictions(y, pred)
            rows.append({"model": name, **metrics})
        except Exception:
            logger.exception("Model %s failed during cross-validation", name)
            raise

    results = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    results["is_best"] = False
    if not results.empty:
        results.loc[0, "is_best"] = True
        best_mae = float(results.loc[0, "mae"])
        results["mae_delta"] = results["mae"] - best_mae
    return results, predictions


def best_model_name(comparison: pd.DataFrame) -> str:
    """Return the lowest-MAE model name from a comparison table."""
    return str(comparison.iloc[0]["model"])


def resolve_model_name(model: str | None, comparison: pd.DataFrame | None = None) -> str:
    """Pick the model to fit: explicit CLI choice, else CV best.

    Args:
        model (str | None): User-supplied model name from --model.
        comparison (pd.DataFrame | None): Fresh or cached benchmark table.

    Returns:
        str: A registered MODEL_BUILDERS key.

    Raises:
        ValueError: If model is not a registered name.
    """
    if model is not None:
        if model not in MODEL_BUILDERS:
            msg = f"Unknown model {model!r}. Choices: {', '.join(MODEL_BUILDERS)}"
            raise ValueError(msg)
        return model
    if comparison is not None and not comparison.empty:
        return best_model_name(comparison)
    msg = "No model specified and no comparison table available."
    raise ValueError(msg)


def build_model_by_name(name: str) -> Pipeline:
    """Instantiate a pipeline for a registered model name."""
    builder = MODEL_BUILDERS[name]
    return builder()


def fit_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, save_path: str | Path) -> Pipeline:
    """Fit a pipeline on all data and persist it with joblib."""
    pipe.fit(X, y)
    joblib.dump(pipe, save_path)
    return pipe
