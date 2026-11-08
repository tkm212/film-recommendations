"""Multi-page Dash dashboard for the film rating model.

The public entry point is create_app, which takes the films-with-predictions
DataFrame produced by the training pipeline and returns a ready-to-run Dash
app with four pages: Overview (/), Films (/films), Model (/model) and About (/about).

Cross-validated model benchmarks live in a separate comparison dashboard
(scripts/compare.py → film_recommendations.comparison_dashboard.create_comparison_app).

The building blocks (theme tokens, components, figures, data helpers) are
importable individually for composing new dashboards:

    from film_recommendations.dashboard import components, data, figures, theme
"""

import pandas as pd
from dash import Dash, html, page_container

from . import data
from .components import navbar
from .theme import FONT_FAMILY, FONT_URL


def create_app(
    films_df: pd.DataFrame,
    title: str = "Film Recommendations",
    importance: dict[str, pd.DataFrame] | None = None,
    selected_model: str | None = None,
    taste_films: pd.DataFrame | None = None,
) -> Dash:
    """Create the multi-page dashboard app.

    Args:
        films_df (pd.DataFrame): Rated films with out-of-fold pred and err columns.
        title (str): Browser title. Defaults to "Film Recommendations".
        importance (dict[str, pd.DataFrame] | None): Feature-importance tables
            from analyze_model, for the Model page.
        selected_model (str | None): Name of the fitted pipeline shown on Overview.
        taste_films (pd.DataFrame | None): Full taste-map universe (watched.csv) with in_library flag.

    Returns:
        Dash: The configured Dash application.
    """
    data.set_films(films_df)
    data.set_taste_films(taste_films)
    data.set_importance(importance)
    data.set_selected_model(selected_model)

    app = Dash(
        __name__,
        use_pages=True,
        pages_folder="",
        title=title,
        external_stylesheets=[FONT_URL],
        suppress_callback_exceptions=True,
    )

    # Imported after app creation so dash.register_page can attach to this app.
    from .pages import about, films, model, overview, taste_map  # noqa: F401

    app.layout = html.Div(
        style={"fontFamily": FONT_FAMILY},
        children=[
            navbar(),
            html.Main(page_container, className="page-container"),
        ],
    )
    return app
