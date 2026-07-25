"""Train, predict, and evaluate the Letterboxd rating model."""

import logging

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import (
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL_PATH,
    DEFAULT_SNAP_STEP,
    DEFAULT_TEST_SIZE,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RATING_CLIP_MAX,
    RATING_CLIP_MIN,
    RIDGE_ALPHA,
    SPARSE_THRESHOLD,
    TARGET_COLUMN,
    TEXT_FEATURE_COLUMNS,
    TFIDF_KEYWORDS_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_OVERVIEW_MAX_FEATURES,
    TFIDF_PEOPLE_MAX_FEATURES,
    TFIDF_PIPE_MAX_FEATURES,
)

logger = logging.getLogger(__name__)


def _split_on_pipe(s: str) -> list[str]:
    """Tokenize a pipe-separated string (module-level so the pipeline is picklable)."""
    return s.split("|")


def _identity(s: str) -> str:
    """Pass-through preprocessor (module-level so the pipeline is picklable)."""
    return s


def create_transformed_features(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build derived numeric/text features and extract target for modeling.

    Args:
        features_df (pd.DataFrame): DataFrame with TMDB-derived columns
            (budget, revenue, vote_count, genres, production_countries,
            overview, keywords, people_text, original_language,
            release_year_tmdb) and Rating.

    Returns:
        tuple[pd.DataFrame, pd.Series]: (data, y). data is the DataFrame
        with added columns (log_budget, log_revenue, log_vote_count,
        genres_count, countries_count) and filled NaNs for text fields;
        y is the Rating target as a Series.
    """
    data = features_df.copy()

    # Derived numeric features
    data["log_budget"] = np.log1p(data["budget"].fillna(0))
    data["log_revenue"] = np.log1p(data["revenue"].fillna(0))
    data["log_vote_count"] = np.log1p(data["vote_count"].fillna(0))
    data["genres_count"] = data["genres"].fillna("").apply(lambda s: 0 if s == "" else len(s.split("|")))
    data["countries_count"] = (
        data["production_countries"].fillna("").apply(lambda s: 0 if s == "" else len(s.split("|")))
    )

    # Fill missing for text fields
    for col in TEXT_FEATURE_COLUMNS:
        data[col] = data[col].fillna("")

    # Target
    y = data[TARGET_COLUMN].astype(float)

    return data, y


def split_train_test(
    data: pd.DataFrame, test_size: float = DEFAULT_TEST_SIZE
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split data into train and test by time (last test_size fraction as holdout).

    Args:
        data (pd.DataFrame): DataFrame sorted by Date with feature columns
            and a Rating column.
        test_size (float): Fraction of rows to use as test set. Defaults to
            DEFAULT_TEST_SIZE.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: (X_train,
        y_train, X_test, y_test) of feature DataFrames and target Series.
    """
    # This can definitely be rewritten using sklearn
    data = data.sort_values("Date").reset_index(drop=True)

    train_size = 1 - test_size
    cut = int(len(data) * train_size)
    train_df = data.iloc[:cut].copy()
    test_df = data.iloc[cut:].copy()

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(float)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(float)

    logger.info("Train rows: %d | Test rows: %d", len(train_df), len(test_df))

    return X_train, y_train, X_test, y_test


def fit_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_path: str = DEFAULT_MODEL_PATH,
) -> Pipeline:
    """Fit a Ridge regression pipeline (preprocess + model) and save to disk.

    Args:
        X_train (pd.DataFrame): Training feature DataFrame (numeric,
            categorical, text columns).
        y_train (pd.Series): Training target Series (Rating).
        save_path (str): Path to save the fitted pipeline with joblib.
            Defaults to DEFAULT_MODEL_PATH.

    Returns:
        Pipeline: The fitted sklearn Pipeline (preprocess + Ridge).
    """
    text_overview = TfidfVectorizer(
        max_features=TFIDF_OVERVIEW_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
    )

    text_keywords = TfidfVectorizer(
        max_features=TFIDF_KEYWORDS_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
    )

    text_people = TfidfVectorizer(
        max_features=TFIDF_PEOPLE_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
    )

    # Treat pipe-separated as space-separated tokens, then TF-IDF
    pipe_as_text = TfidfVectorizer(
        max_features=TFIDF_PIPE_MAX_FEATURES,
        tokenizer=_split_on_pipe,
        preprocessor=_identity,
        token_pattern=None,
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", StandardScaler(with_mean=False)),  # works with sparse output too
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            ("lang", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("overview", text_overview, "overview"),
            ("keywords", text_keywords, "keywords"),
            ("people", text_people, "people_text"),
            ("genres", pipe_as_text, "genres"),
            ("countries", pipe_as_text, "production_countries"),
        ],
        remainder="drop",
        sparse_threshold=SPARSE_THRESHOLD,
    )

    model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    pipe.fit(X_train, y_train)
    joblib.dump(pipe, save_path)
    return pipe


def make_predictions(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate predictions with a fitted pipeline and clip to [0.5, 5.0].

    Args:
        pipe (Pipeline): Fitted sklearn Pipeline (e.g. from fit_pipeline).
        X_test (pd.DataFrame): Test feature DataFrame.
        y_test (pd.Series): Test target Series (used for eval_df and logging).

    Returns:
        tuple[pd.DataFrame, np.ndarray]: (eval_df, pred_clipped). eval_df has
        columns Date, Name, Year, Rating, pred, err; pred_clipped is the
        clipped prediction array.
    """
    pred = pipe.predict(X_test)
    pred_clipped = np.clip(pred, RATING_CLIP_MIN, RATING_CLIP_MAX)

    mae = mean_absolute_error(y_test, pred_clipped)
    rmse = mean_squared_error(y_test, pred_clipped)
    r2 = r2_score(y_test, pred_clipped)

    logging.info((mae, rmse, r2))

    test_df = X_test.assign(Rating=y_test)
    eval_df = test_df[["Date", "Name", "Year", "Rating"]].copy()
    eval_df["pred"] = pred_clipped
    eval_df["err"] = eval_df["pred"] - eval_df["Rating"]
    eval_df.sort_values("err").head(10), eval_df.sort_values("err").tail(10)

    spearman = spearmanr(y_test, pred_clipped).correlation
    logging.info(spearman)

    kendall = kendalltau(y_test, pred_clipped).correlation
    logging.info(kendall)

    return eval_df, pred_clipped


def evaluate_rating_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    eval_df: pd.DataFrame,
    snap_step: float = DEFAULT_SNAP_STEP,
) -> dict[str, object]:
    """Evaluate snapped rating predictions with multiple metrics.

    Args:
        y_true (pd.Series): True ratings.
        y_pred (np.ndarray): Raw predicted ratings (after clipping,
            before snapping).
        eval_df (pd.DataFrame): DataFrame containing at least columns Rating
            and err (used for bucketed MAE by rating).
        snap_step (float): Rating granularity in stars. Defaults to
            DEFAULT_SNAP_STEP.

    Returns:
        dict[str, object]: Keys pred_snapped, exact_match_rate,
        within_half_star, bucket_mae_by_rating.
    """
    # Snap predictions to nearest step (e.g., 0.5 stars)
    pred_snapped = np.round(y_pred / snap_step) * snap_step

    # Metrics
    exact_match_rate = (pred_snapped == y_true).mean()
    within_half_star = (np.abs(pred_snapped - y_true) <= snap_step).mean()

    logger.info("Exact match rate: %.4f", exact_match_rate)
    logger.info("Within %.1f-star accuracy: %.4f", snap_step, within_half_star)

    # Bucketed absolute error by true rating
    bucket_eval = eval_df.assign(abs_err=lambda d: np.abs(d["err"])).groupby("Rating")["abs_err"].mean()

    logger.debug("Bucketed MAE by rating:\n%s", bucket_eval)

    return {
        "pred_snapped": pred_snapped,
        "exact_match_rate": exact_match_rate,
        "within_half_star": within_half_star,
        "bucket_mae_by_rating": bucket_eval,
    }
