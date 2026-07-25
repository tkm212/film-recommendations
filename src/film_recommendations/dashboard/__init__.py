"""Multi-page Dash dashboard for the film rating model.

The public entry point is create_app, which takes the films-with-predictions
DataFrame produced by the training pipeline and returns a ready-to-run Dash
app with three pages: Overview (/), Films (/films) and About (/about).

The building blocks (theme tokens, components, figures, data helpers) are
importable individually for composing new dashboards:

    from film_recommendations.dashboard import components, data, figures, theme
"""

import pandas as pd
from dash import Dash, html, page_container

from . import data
from .components import navbar
from .theme import FONT_FAMILY, FONT_URL


def create_app(films_df: pd.DataFrame, title: str = "Film Recommendations") -> Dash:
    """Create the multi-page dashboard app.

    Args:
        films_df (pd.DataFrame): All rated films with feature columns plus
            pred, err and split ("train"/"test").
        title (str): Browser title. Defaults to "Film Recommendations".

    Returns:
        Dash: The configured Dash application.
    """
    data.set_films(films_df)

    app = Dash(
        __name__,
        use_pages=True,
        pages_folder="",
        title=title,
        external_stylesheets=[FONT_URL],
        suppress_callback_exceptions=True,
    )

    # Imported after app creation so dash.register_page can attach to this app.
    from .pages import about, films, overview  # noqa: F401

    app.layout = html.Div(
        style={"fontFamily": FONT_FAMILY},
        children=[
            navbar(),
            html.Main(page_container, className="page-container"),
        ],
    )
    return app
