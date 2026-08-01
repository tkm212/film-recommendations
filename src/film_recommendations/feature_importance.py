"""Extract interpretable feature importance from the fitted Ridge pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

GROUP_LABELS = {
    "num": "Numeric metadata",
    "lang": "Original language",
    "overview": "Plot overview (TF-IDF)",
    "keywords": "Keywords (TF-IDF)",
    "people": "Cast & director (TF-IDF)",
    "genres": "Genres",
    "countries": "Production countries",
    "emb": "Sentence embeddings (PCA)",
}

# ColumnTransformer transformer names mapped to raw input columns for group drops.
GROUP_COLUMNS: dict[str, list[str]] = {
    "num": [
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
    ],
    "lang": ["original_language"],
    "overview": ["overview"],
    "keywords": ["keywords"],
    "people": ["people_text"],
    "genres": ["genres"],
    "countries": ["production_countries"],
}


def _feature_group(feature_name: str) -> str:
    """Map a ColumnTransformer output name to its transformer group key."""
    return feature_name.split("__", 1)[0]


def _display_group(group_key: str) -> str:
    return GROUP_LABELS.get(group_key, group_key)


def coefficient_importance(pipe: Pipeline) -> pd.DataFrame:
    """Build a per-feature importance table from Ridge regression coefficients.

    Args:
        pipe (Pipeline): Fitted preprocess + Ridge pipeline from fit_pipeline.

    Returns:
        pd.DataFrame: Columns feature, group, group_label, coefficient,
        abs_coefficient, direction.
    """
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    names = preprocess.get_feature_names_out()
    coefs = np.asarray(model.coef_).ravel()

    rows = []
    for name, coef in zip(names, coefs, strict=True):
        group = _feature_group(name)
        feature = name.split("__", 1)[1] if "__" in name else name
        rows.append({
            "feature": feature,
            "group": group,
            "group_label": _display_group(group),
            "coefficient": float(coef),
            "abs_coefficient": float(abs(coef)),
            "direction": "higher rating" if coef > 0 else "lower rating",
        })

    return pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def group_importance(coef_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate absolute coefficients by feature group.

    Args:
        coef_df (pd.DataFrame): Output of coefficient_importance.

    Returns:
        pd.DataFrame: Columns group, group_label, total_abs_coefficient,
        share (fraction of total abs mass).
    """
    grouped = (
        coef_df.groupby(["group", "group_label"], as_index=False)["abs_coefficient"]
        .sum()
        .rename(columns={"abs_coefficient": "total_abs_coefficient"})
        .sort_values("total_abs_coefficient", ascending=False)
    )
    total = grouped["total_abs_coefficient"].sum()
    grouped["share"] = grouped["total_abs_coefficient"] / total if total else 0.0
    return grouped.reset_index(drop=True)


def _group_columns_for(X: pd.DataFrame) -> dict[str, list[str]]:
    """Feature-group column map, including sentence-embedding columns when present."""
    groups = dict(GROUP_COLUMNS)
    emb_cols = [col for col in X.columns if col.startswith("emb_")]
    if emb_cols:
        groups["emb"] = emb_cols
    return groups


def permutation_group_importance(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance by dropping entire feature groups (MAE increase).

    Shuffling all columns in a group measures how much the model relies on that
    block of inputs, complementing coefficient-based importance for TF-IDF blocks.

    Args:
        pipe (Pipeline): Fitted pipeline.
        X (pd.DataFrame): Feature DataFrame used to fit the model.
        y (pd.Series): Target ratings.
        n_repeats (int): Permutation repeats per group. Defaults to 5.
        random_state (int): Random seed. Defaults to 42.

    Returns:
        pd.DataFrame: Columns group, group_label, importance_mean, importance_std.
    """
    rng = np.random.default_rng(random_state)
    baseline = pipe.predict(X)
    baseline_mae = np.mean(np.abs(baseline - y))

    rows = []
    for group, columns in _group_columns_for(X).items():
        present = [col for col in columns if col in X.columns]
        if not present:
            continue

        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            for col in present:
                X_perm[col] = rng.permutation(X_perm[col].values)
            perm_mae = np.mean(np.abs(pipe.predict(X_perm) - y))
            scores.append(perm_mae - baseline_mae)

        rows.append({
            "group": group,
            "group_label": _display_group(group),
            "importance_mean": float(np.mean(scores)),
            "importance_std": float(np.std(scores)),
        })

    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def analyze_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, pd.DataFrame] | None:
    """Run coefficient and group-level permutation importance analysis.

    Args:
        pipe (Pipeline): Fitted pipeline (Ridge-based only; tree models return None).
        X (pd.DataFrame): Training features (same schema as model input).
        y (pd.Series): Target ratings.

    Returns:
        dict[str, pd.DataFrame] | None: Keys coefficients, groups, permutation_groups,
        or None when the model has no linear coefficients.
    """
    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        return None

    coef_df = coefficient_importance(pipe)
    return {
        "coefficients": coef_df,
        "groups": group_importance(coef_df),
        "permutation_groups": permutation_group_importance(pipe, X, y),
    }
