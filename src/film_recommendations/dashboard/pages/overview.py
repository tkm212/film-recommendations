"""Overview page: model metrics and charts with cross-filtering controls."""

import dash
import pandas as pd
from dash import Input, Output, callback, dcc, html
from dash.development.base_component import Component

from .. import data
from ..components import card, data_table, metric_card, metric_row, page_header, section_title
from ..figures import empty_figure, error_histogram_figure, mae_by_rating_figure, predicted_vs_actual_figure
from ..theme import COLORS

MISS_COLUMNS = [
    {"name": "Watched", "id": "Date"},
    {"name": "Film", "id": "Name"},
    {"name": "Year", "id": "Year"},
    {"name": "Genres", "id": "genres_display"},
    {"name": "Language", "id": "language_display"},
    {"name": "Runtime", "id": "runtime_display"},
    {"name": "Your rating", "id": "Rating"},
    {"name": "Predicted", "id": "pred"},
    {"name": "Error", "id": "err"},
]


def _toolbar_item(label: str, component: Component, extra_class: str = "") -> html.Div:
    return html.Div(
        className=f"toolbar-item {extra_class}".strip(),
        children=[html.Span(label, className="toolbar-label"), component],
    )


def layout() -> html.Div:
    """Build the overview page layout with filter controls and placeholders."""
    year_min, year_max = data.year_bounds()
    model_name = data.get_selected_model()
    subtitle = (
        f"Out-of-fold predictions from {model_name} (3-fold CV). "
        "Every rating below was predicted by a model that never saw that film during training."
        if model_name
        else "Out-of-fold predictions from 3-fold cross-validation — "
        "every rating below was predicted by a model that never saw that film during training."
    )

    toolbar = html.Div(
        className="toolbar",
        children=[
            _toolbar_item(
                "Genre",
                dcc.Dropdown(id="ov-genre", options=data.genre_options(), multi=True, placeholder="All"),
                "toolbar-dd",
            ),
            _toolbar_item(
                "Language",
                dcc.Dropdown(id="ov-lang", options=data.language_options(), multi=True, placeholder="All"),
                "toolbar-dd",
            ),
            _toolbar_item(
                "Year watched",
                dcc.RangeSlider(
                    id="ov-years",
                    min=year_min,
                    max=year_max,
                    step=1,
                    value=[year_min, year_max],
                    marks=None,
                    tooltip={"placement": "bottom"},
                    allowCross=False,
                ),
                "toolbar-slider",
            ),
            _toolbar_item(
                "Your rating",
                dcc.RangeSlider(
                    id="ov-rating",
                    min=0.5,
                    max=5.0,
                    step=0.5,
                    value=[0.5, 5.0],
                    marks=None,
                    tooltip={"placement": "bottom"},
                    allowCross=False,
                ),
                "toolbar-slider",
            ),
            html.Div(
                className="toolbar-meta",
                children=[
                    html.Span(id="ov-count", className="filter-count"),
                    html.Span(id="ov-selected", className="filter-selected"),
                ],
            ),
        ],
    )

    return html.Div([
        toolbar,
        page_header(
            "Model overview",
            subtitle,
        ),
        html.Div(id="ov-metrics"),
        html.Div(
            className="graph-grid-2",
            children=[
                card([dcc.Graph(id="ov-scatter", config={"displayModeBar": False})]),
                card([dcc.Graph(id="ov-hist", config={"displayModeBar": False})]),
            ],
        ),
        card([dcc.Graph(id="ov-mae", config={"displayModeBar": False})]),
        html.Div(
            children=[
                section_title("Out-of-fold predictions"),
                card([data_table("ov-table", MISS_COLUMNS, page_size=10, sortable=True)]),
            ],
        ),
    ])


def _prediction_records(df: pd.DataFrame) -> list[dict]:
    """All filtered rows formatted for the table, worst misses first."""
    display = df[["Date", "Name", "Year", "Rating", "pred", "err"]].copy()
    display["Date"] = display["Date"].dt.date.astype(str)
    display["genres_display"] = df["genres"].str.replace("|", ", ", regex=False)
    display["language_display"] = df["original_language"].map(data.language_name)
    display["runtime_display"] = df["runtime"].map(lambda r: f"{int(r)} min" if pd.notna(r) and r else "")
    display["pred"] = display["pred"].round(2)
    display["err"] = display["err"].round(2)
    return display.reindex(display["err"].abs().sort_values(ascending=False).index).to_dict("records")


def _metric_cards(df: pd.DataFrame) -> html.Div:
    """Metric card row for the filtered films (placeholders if too small)."""
    if len(df) < 2:
        values = dict.fromkeys(("mae", "rmse", "r2", "spearman", "exact", "half"), "—")
    else:
        m = data.compute_metrics(df)
        values = {
            "mae": f"{m['mae']:.3f}",
            "rmse": f"{m['rmse']:.3f}",
            "r2": f"{m['r2']:.3f}",
            "spearman": f"{m['spearman']:.3f}",
            "exact": f"{m['exact_match_rate']:.0%}",
            "half": f"{m['within_half_star']:.0%}",
        }
    return metric_row([
        metric_card("MAE", values["mae"], "mean absolute error"),
        metric_card("RMSE", values["rmse"], "root mean squared error"),
        metric_card("R²", values["r2"], "variance explained"),
        metric_card("Spearman", values["spearman"], "rank correlation"),
        metric_card("Exact match", values["exact"], "nearest half star"),
        metric_card("Within half star", values["half"], "of true rating"),
    ])


def _selected_film(click_data: dict | None, df: pd.DataFrame) -> tuple[str | None, str]:
    """Resolve a scatter click into (film name, summary text)."""
    if not click_data or not click_data.get("points"):
        return None, ""
    customdata = click_data["points"][0].get("customdata")
    if not customdata:
        return None, ""

    name = customdata[0]
    rows = df[df["Name"] == name]
    if rows.empty:
        return None, ""
    row = rows.iloc[0]
    return name, f"Selected: {name} ({row['Year']}) — you rated {row['Rating']:.1f}, model predicted {row['pred']:.2f}"


def _highlight_styles(name: str | None) -> list[dict]:
    """Conditional styles: error coloring plus highlight for a selected film."""
    from ..components import ERR_COLOR_STYLES

    styles = list(ERR_COLOR_STYLES)
    if name:
        safe = name.replace('"', '\\"')
        styles.append({
            "if": {"filter_query": f'{{Name}} = "{safe}"'},
            "backgroundColor": COLORS["primary_soft"],
            "fontWeight": "600",
        })
    return styles


@callback(
    Output("ov-metrics", "children"),
    Output("ov-scatter", "figure"),
    Output("ov-hist", "figure"),
    Output("ov-mae", "figure"),
    Output("ov-table", "data"),
    Output("ov-table", "style_data_conditional"),
    Output("ov-count", "children"),
    Output("ov-selected", "children"),
    Input("ov-genre", "value"),
    Input("ov-lang", "value"),
    Input("ov-years", "value"),
    Input("ov-rating", "value"),
    Input("ov-scatter", "clickData"),
)
def update_overview(genres, languages, years, ratings, click_data):
    """Re-render all overview charts and the predictions table from the active filters."""
    films = data.get_films()
    df = data.filter_films(films, genres, languages, years, ratings)

    count_text = f"Showing {len(df)} of {len(films)} films"

    if df.empty:
        empty = empty_figure()
        return _metric_cards(df), empty, empty, empty, [], _highlight_styles(None), count_text, ""

    selected_name, selected_text = _selected_film(click_data, df)

    return (
        _metric_cards(df),
        predicted_vs_actual_figure(df),
        error_histogram_figure(df),
        mae_by_rating_figure(df),
        _prediction_records(df),
        _highlight_styles(selected_name),
        count_text,
        selected_text,
    )


# Registered after layout is defined so dash.page_registry picks it up.
dash.register_page(__name__, path="/", name="Overview", title="Film Recommendations — Overview", layout=layout)
