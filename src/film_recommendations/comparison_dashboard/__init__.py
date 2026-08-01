"""Standalone Dash dashboard for cross-validated model benchmarks."""

import pandas as pd
from dash import Dash, dcc, html

from film_recommendations.dashboard.components import card, data_table, page_header, section_title
from film_recommendations.dashboard.figures import model_comparison_figure
from film_recommendations.dashboard.theme import FONT_FAMILY, FONT_URL

COMPARISON_COLUMNS = [
    {"name": "Model", "id": "model"},
    {"name": "MAE", "id": "mae", "type": "numeric", "format": {"specifier": ".3f"}},
    {"name": "RMSE", "id": "rmse", "type": "numeric", "format": {"specifier": ".3f"}},
    {"name": "R²", "id": "r2", "type": "numeric", "format": {"specifier": ".3f"}},
    {"name": "Exact match", "id": "exact_match_rate", "type": "numeric", "format": {"specifier": ".1%"}},
    {"name": "Within 0.5★", "id": "within_half_star", "type": "numeric", "format": {"specifier": ".1%"}},
    {"name": "Δ MAE vs best", "id": "mae_delta", "type": "numeric", "format": {"specifier": "+.3f"}},
    {"name": "CV best", "id": "cv_best"},
    {"name": "Deployed", "id": "deployed"},
]


def _comparison_table_data(comparison: pd.DataFrame, selected_model: str | None) -> list[dict]:
    table = comparison.copy()
    table["cv_best"] = table["is_best"].map({True: "Yes", False: ""})
    table["deployed"] = table["model"].eq(selected_model).map({True: "Yes", False: ""}) if selected_model else ""
    return table.to_dict("records")


def create_comparison_app(
    comparison: pd.DataFrame,
    selected_model: str | None = None,
    title: str = "Film Recommendations — Model comparison",
) -> Dash:
    """Create a single-page dashboard for the CV model benchmark table."""
    subtitle = (
        f"3-fold out-of-fold MAE for each candidate. Saved pipeline: {selected_model}."
        if selected_model
        else "3-fold out-of-fold MAE for each candidate. Lower is better."
    )

    app = Dash(__name__, title=title, external_stylesheets=[FONT_URL], suppress_callback_exceptions=True)
    app.layout = html.Div(
        style={"fontFamily": FONT_FAMILY},
        className="page-container",
        children=[
            html.Header(
                className="navbar",
                children=[
                    html.Div(
                        className="nav-brand",
                        children=[
                            html.Span(className="nav-logo", children="FR"),
                            html.Span("Model comparison", className="nav-title"),
                        ],
                    ),
                ],
            ),
            html.Main([
                page_header("Model comparison", subtitle),
                card([dcc.Graph(id="cmp-chart", figure=model_comparison_figure(comparison))]),
                section_title("Cross-validated metrics"),
                card([
                    data_table(
                        "cmp-table",
                        COMPARISON_COLUMNS,
                        page_size=10,
                        sortable=True,
                        data=_comparison_table_data(comparison, selected_model),
                    ),
                ]),
            ]),
        ],
    )
    return app
