"""Feature importance for the saved rating model."""

import dash
import pandas as pd
from dash import Input, Output, callback, dcc, html

from .. import data
from ..components import card, data_table, page_header, section_title
from ..figures import group_coefficient_figure, group_permutation_figure, top_coefficients_figure

FEATURE_COLUMNS = [
    {"name": "Feature", "id": "feature"},
    {"name": "Group", "id": "group_label"},
    {"name": "Coefficient", "id": "coefficient", "type": "numeric", "format": {"specifier": ".4f"}},
    {"name": "|Coefficient|", "id": "abs_coefficient", "type": "numeric", "format": {"specifier": ".4f"}},
    {"name": "Direction", "id": "direction"},
]


def _group_options(coef_df: pd.DataFrame) -> list[dict]:
    options = [{"label": "All groups", "value": "all"}]
    for _, row in coef_df[["group", "group_label"]].drop_duplicates().iterrows():
        options.append({"label": row["group_label"], "value": row["group"]})
    return options


def layout() -> html.Div:
    """Build the feature-importance page for the saved model."""
    model_label = data.get_selected_model() or "saved model"
    sections: list = [
        page_header(
            "Feature importance",
            f"Coefficient and permutation analysis for the {model_label} pipeline "
            "(linear Ridge models only). Run scripts/compare.py to view the CV benchmark.",
        ),
    ]

    if not data.has_importance():
        sections.append(
            card([
                html.P(
                    "Coefficient and permutation importance are available for linear (Ridge) models. "
                    "Retrain with --train if this section is missing.",
                    className="muted-text",
                ),
            ]),
        )
        return html.Div(sections)

    importance = data.get_importance()
    coef_df = importance["coefficients"]

    sections.extend([
        html.Div(
            className="graph-grid-2",
            children=[
                card([dcc.Graph(id="md-group-coef", figure=group_coefficient_figure(importance["groups"]))]),
                card([
                    dcc.Graph(id="md-group-perm", figure=group_permutation_figure(importance["permutation_groups"]))
                ]),
            ],
        ),
        card([
            html.Div(
                className="filter-grid",
                style={"marginBottom": "16px"},
                children=[
                    html.Div([
                        html.Label("Feature group", className="control-label"),
                        dcc.Dropdown(
                            id="md-group-filter",
                            options=_group_options(coef_df),
                            value="all",
                            clearable=False,
                        ),
                    ]),
                ],
            ),
            dcc.Graph(id="md-top-coef", figure=top_coefficients_figure(coef_df)),
        ]),
        section_title("All features ranked by |coefficient|"),
        card([data_table("md-feature-table", FEATURE_COLUMNS, page_size=15, sortable=True)]),
    ])

    return html.Div(sections)


@callback(
    Output("md-top-coef", "figure"),
    Output("md-feature-table", "data"),
    Input("md-group-filter", "value"),
)
def update_feature_views(group: str):
    """Filter top-coefficient chart and feature table by group."""
    if not data.has_importance():
        return top_coefficients_figure(pd.DataFrame()), []

    coef_df = data.get_importance()["coefficients"]
    filtered = coef_df if group == "all" else coef_df[coef_df["group"] == group]
    table_df = filtered.copy()
    table_df["coefficient"] = table_df["coefficient"].round(4)
    table_df["abs_coefficient"] = table_df["abs_coefficient"].round(4)
    return top_coefficients_figure(coef_df, group=None if group == "all" else group), table_df.to_dict("records")


dash.register_page(__name__, path="/model", name="Model", title="Film Recommendations — Model", layout=layout)
