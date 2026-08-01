"""Model pipeline builders for rating prediction benchmarks."""

from collections.abc import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import (
    CATEGORICAL_FEATURES,
    EMBEDDING_PCA_COMPONENTS,
    HGB_SVD_COMPONENTS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RIDGE_ALPHA,
    SPARSE_THRESHOLD,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)
from .embeddings import embedding_column_names
from .ml_fitting import build_pipeline as build_ridge_tfidf_pipeline


class AdaptiveTruncatedSVD(BaseEstimator, TransformerMixin):
    """TruncatedSVD that caps components at min(requested, n_features - 1)."""

    def __init__(self, n_components: int = HGB_SVD_COMPONENTS, random_state: int = RANDOM_STATE) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.svd_: TruncatedSVD | None = None

    def fit(self, X, y=None):
        n = min(self.n_components, max(X.shape[1] - 1, 1))
        self.svd_ = TruncatedSVD(n_components=n, random_state=self.random_state)
        self.svd_.fit(X)
        return self

    def transform(self, X):
        if self.svd_ is None:
            msg = "AdaptiveTruncatedSVD has not been fitted yet."
            raise RuntimeError(msg)
        return self.svd_.transform(X)


class AdaptivePCA(BaseEstimator, TransformerMixin):
    """PCA that caps components at min(requested, n_samples, n_features)."""

    def __init__(self, n_components: int = EMBEDDING_PCA_COMPONENTS, random_state: int = RANDOM_STATE) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.pca_: PCA | None = None

    def fit(self, X, y=None):
        n = min(self.n_components, X.shape[0], X.shape[1])
        self.pca_ = PCA(n_components=n, random_state=self.random_state)
        self.pca_.fit(X)
        return self

    def transform(self, X):
        if self.pca_ is None:
            msg = "AdaptivePCA has not been fitted yet."
            raise RuntimeError(msg)
        return self.pca_.transform(X)


def _embedding_branch(n_pca: int = EMBEDDING_PCA_COMPONENTS) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", AdaptivePCA(n_components=n_pca, random_state=RANDOM_STATE)),
    ])


def build_ridge_metadata_pipeline() -> Pipeline:
    """Ridge regression on numeric metadata and language only (no text)."""
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("lang", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=SPARSE_THRESHOLD,
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("model", Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)),
    ])


def build_ridge_embedding_pipeline(
    emb_cols: list[str] | None = None,
    n_pca: int = EMBEDDING_PCA_COMPONENTS,
) -> Pipeline:
    """Ridge regression on numeric, language, and PCA-compressed sentence embeddings."""
    emb_cols = emb_cols or embedding_column_names()
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("lang", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("emb", _embedding_branch(n_pca), emb_cols),
        ],
        remainder="drop",
        sparse_threshold=SPARSE_THRESHOLD,
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("model", Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)),
    ])


def build_ridge_hybrid_pipeline(
    emb_cols: list[str] | None = None,
    n_pca: int = EMBEDDING_PCA_COMPONENTS,
) -> Pipeline:
    """Ridge on numeric, language, TF-IDF text, and PCA-compressed sentence embeddings."""
    emb_cols = emb_cols or embedding_column_names()
    base_preprocess = build_ridge_tfidf_pipeline().named_steps["preprocess"]
    transformers = [
        *base_preprocess.transformers,
        ("emb", _embedding_branch(n_pca), emb_cols),
    ]
    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder=base_preprocess.remainder,
        sparse_threshold=base_preprocess.sparse_threshold,
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("model", Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)),
    ])


def build_hgb_hybrid_pipeline(
    emb_cols: list[str] | None = None,
    n_pca: int = EMBEDDING_PCA_COMPONENTS,
    svd_dim: int = HGB_SVD_COMPONENTS,
) -> Pipeline:
    """Gradient boosting on dense numeric, language, SVD(TF-IDF), and PCA(embeddings)."""
    emb_cols = emb_cols or embedding_column_names()
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("lang", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("emb", _embedding_branch(n_pca), emb_cols),
            (
                "txt",
                Pipeline([
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=3000,
                            ngram_range=TFIDF_NGRAM_RANGE,
                            min_df=TFIDF_MIN_DF,
                        ),
                    ),
                    ("svd", AdaptiveTruncatedSVD(n_components=svd_dim, random_state=RANDOM_STATE)),
                ]),
                "merged_text",
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    model = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.05,
        max_iter=500,
        min_samples_leaf=8,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


MODEL_BUILDERS: dict[str, Callable[[], Pipeline]] = {
    "Ridge (metadata only)": build_ridge_metadata_pipeline,
    "Ridge + TF-IDF (baseline)": build_ridge_tfidf_pipeline,
    "Ridge + sentence embeddings": build_ridge_embedding_pipeline,
    "Ridge + TF-IDF + embeddings": build_ridge_hybrid_pipeline,
    "Gradient boosting + TF-IDF + embeddings": build_hgb_hybrid_pipeline,
}
