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

logger = logging.getLogger(__name__)


def create_transformed_features(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
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
    for col in ["overview", "keywords", "people_text", "genres", "production_countries", "original_language"]:
        data[col] = data[col].fillna("")

    # Target
    y = data["Rating"].astype(float)

    return data, y


def split_train_test(
    data: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    # This can definitely be rewritten using sklearn
    data = data.sort_values("Date").reset_index(drop=True)

    # Use last 20% as holdout
    train_size = 1 - test_size
    cut = int(len(data) * train_size)
    train_df = data.iloc[:cut].copy()
    test_df = data.iloc[cut:].copy()

    TARGET = "Rating"

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(float)

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(float)

    logging.info(len(train_df), len(test_df))

    return X_train, y_train, X_test, y_test


def fit_pipeline(
    X_train: pd.DataFrame, y_train: pd.Series, save_path: str = "./letterboxd_tmdb_rating_model.joblib"
) -> Pipeline:

    text_overview = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=2)

    text_keywords = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), min_df=2)

    text_people = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), min_df=2)

    # Treat pipe-separated as space-separated tokens, then TF-IDF
    pipe_as_text = TfidfVectorizer(
        max_features=800, tokenizer=lambda s: s.split("|"), preprocessor=lambda s: s, token_pattern=None
    )

    numeric_features = [
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
    ]

    categorical_features = ["original_language"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", StandardScaler(with_mean=False)),  # works with sparse output too
                    ]
                ),
                numeric_features,
            ),
            ("lang", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("overview", text_overview, "overview"),
            ("keywords", text_keywords, "keywords"),
            ("people", text_people, "people_text"),
            ("genres", pipe_as_text, "genres"),
            ("countries", pipe_as_text, "production_countries"),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = Ridge(alpha=2.0, random_state=42)

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    pipe.fit(X_train, y_train)
    joblib.dump(pipe, "./letterboxd_tmdb_rating_model.joblib")
    return pipe


def make_predictions(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    pred = pipe.predict(X_test)
    pred_clipped = np.clip(pred, 0.5, 5.0)

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
    snap_step: float = 0.5,
) -> dict[str, object]:
    """
    Evaluate snapped rating predictions with multiple metrics.

    :param y_true: True ratings
    :param y_pred: Raw predicted ratings (after clipping, before snapping)
    :param eval_df: DataFrame containing at least columns ["Rating", "err"]
    :param snap_step: Rating granularity (default = 0.5 stars)
    :return: Dictionary of evaluation results
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
