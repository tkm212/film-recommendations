"""About page: methodology and pipeline documentation."""

import dash
from dash import html

from ..components import card, page_header, section_title
from ..data import get_films


def _film_count_phrase() -> str:
    try:
        return f"On this {len(get_films())}-film dataset"
    except RuntimeError:
        return "On this dataset"


def _item(term: str, description: str) -> html.Div:
    return html.Div(
        className="about-item",
        children=[html.Div(term, className="about-term"), html.Div(description, className="about-desc")],
    )


def layout() -> html.Div:
    """Build the static about/methodology page."""
    return html.Div([
        page_header(
            "About this project",
            "Predicting Letterboxd star ratings (0.5 to 5.0) from film metadata, "
            "to benchmark future recommendation models against.",
        ),
        section_title("Data"),
        card([
            _item(
                "Source",
                "Letterboxd exports: ratings.csv for star ratings and modeling; "
                "optional diary.csv for every film you have logged as watched (rated or not).",
            ),
            _item(
                "Enrichment",
                "Each film is matched to The Movie Database (TMDB) and enriched with metadata: "
                "runtime, budget, revenue, popularity, vote statistics, genres, production countries, "
                "keywords, plot overview, director and top-billed cast.",
            ),
            _item(
                "Caching",
                "The enriched feature table is cached to CSV so the model can be retrained "
                "without re-hitting the TMDB API.",
            ),
        ]),
        section_title("Features"),
        card([
            _item(
                "Numeric",
                "Runtime, popularity, vote statistics, log-scaled budget / revenue / vote count, "
                "genre and production-country counts — standardized before fitting.",
            ),
            _item("Categorical", "Original language, one-hot encoded."),
            _item(
                "Text",
                "Plot overview, keywords, and director/cast names via TF-IDF; genres and "
                "production countries tokenized on the pipe separator. Sentence embeddings "
                "(all-MiniLM-L6-v2) capture semantic plot/people text and are PCA-reduced "
                "before modeling.",
            ),
        ]),
        section_title("Model & evaluation"),
        card([
            _item(
                "Candidates",
                "Ridge (metadata only), Ridge + TF-IDF, Ridge + sentence embeddings, "
                "Ridge + TF-IDF + embeddings, and histogram gradient boosting on dense "
                "SVD(TF-IDF) + PCA(embeddings). All use the same 3-fold out-of-fold protocol.",
            ),
            _item(
                "Model selection",
                "Run --train to benchmark every candidate. By default the lowest-MAE model is "
                "fitted and saved; override with --model. View the benchmark table with "
                "scripts/compare.py.",
            ),
            _item(
                "Findings",
                f"{_film_count_phrase()}, sparse TF-IDF text features outperform sentence "
                "embeddings and gradient boosting — likely because taste correlates with specific "
                "keywords and people, and sample size is small for dense text vectors.",
            ),
            _item(
                "Validation",
                "3-fold cross-validation: every film's rating is predicted out-of-fold by a model "
                "that never saw it during training, so all films get a directly comparable prediction. "
                "The saved model is then refit on the full dataset.",
            ),
            _item(
                "Feature importance",
                "The Model page reports Ridge coefficient mass by feature group (numeric, language, "
                "TF-IDF text blocks) and group-level permutation importance (MAE increase when a block "
                "is shuffled), plus the strongest individual coefficients.",
            ),
            _item(
                "Taste map",
                "The Taste map page PCA-reduces sentence embeddings to three dimensions and "
                "clusters films with KMeans so you can explore taste neighborhoods interactively.",
            ),
            _item(
                "Metrics",
                "MAE, RMSE and R² on the out-of-fold predictions (clipped to 0.5 to 5.0), Spearman rank "
                "correlation, and accuracy after snapping predictions to the nearest half star.",
            ),
        ]),
    ])


# Registered after layout is defined so dash.page_registry picks it up.
dash.register_page(__name__, path="/about", name="About", title="Film Recommendations — About", layout=layout)
