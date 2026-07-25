"""About page: methodology and pipeline documentation."""

import dash
from dash import html

from ..components import card, page_header, section_title

dash.register_page(__name__, path="/about", name="About", title="Film Recommendations — About")


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
            _item("Source", "A raw Letterboxd ratings export (date watched, film, year, star rating)."),
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
                "production countries tokenized on the pipe separator.",
            ),
        ]),
        section_title("Model & evaluation"),
        card([
            _item("Model", "Ridge regression over the combined sparse feature matrix (alpha = 2.0)."),
            _item(
                "Split",
                "Chronological: the most recent 20% of ratings are held out as the test set, "
                "mimicking a realistic forecasting setting and avoiding leakage from future ratings.",
            ),
            _item(
                "Metrics",
                "MAE, RMSE and R² on the raw predictions (clipped to 0.5 to 5.0), Spearman rank "
                "correlation, and accuracy after snapping predictions to the nearest half star.",
            ),
        ]),
    ])
