"""Overview page: model metrics and charts with cross-filtering controls."""

import dash
import pandas as pd
from dash import Input, Output, callback, dcc, html

from .. import data
from ..components import card, control, data_table, metric_card, metric_row, page_header, section_title
from ..figures import empty_figure, error_histogram_figure, mae_by_rating_figure, predicted_vs_actual_figure
from ..theme import COLORS

dash.register_page(__name__, path="/", name="Overview", title="Film Recommendations — Overview")

MISS_COLUMNS = [
    {"name": "Watched", "id": "Date"},
    {"name": "Film", "id": "Name"},
    {"name": "Year", "id": "Year"},
    {"name": "Your rating", "id": "Rating"},
    {"name": "Predicted", "id": "pred"},
    {"name": "Error", "id": "err"},
]


def layout() -> html.Div:
    """Build the overview page layout with filter controls and placeholders."""
    year_min, year_max = data.year_bounds()

    filters = card(
        [
            html.Div(
                className="filter-grid",
                children=[
                    control(
                        "Genre",
                        dcc.Dropdown(
                            id="ov-genre",
                            options=data.genre_options(),
                            multi=True,
                            placeholder="All genres",
                        ),
                    ),
                    control(
                        "Language",
                        dcc.Dropdown(
                            id="ov-lang",
                            options=data.language_options(),
                            multi=True,
                            placeholder="All languages",
                        ),
                    ),
                    control(
                        "Year watched",
                        dcc.RangeSlider(
                            id="ov-years",
                            min=year_min,
                            max=year_max,
                            step=1,
                            value=[year_min, year_max],
                            marks={year_min: str(year_min), year_max: str(year_max)},
                            tooltip={"placement": "bottom"},
                            allowCross=False,
                        ),
                    ),
                    control(
                        "Your rating",
                        dcc.RangeSlider(
                            id="ov-rating",
                            min=0.5,
                            max=5.0,
                            step=0.5,
                            value=[0.5, 5.0],
                            marks={0.5: "0.5", 5: "5.0"},
                            tooltip={"placement": "bottom"},
                            allowCross=False,
                        ),
                    ),
                ],
            ),
            html.Div(
                className="filter-footer",
                children=[
                    html.Span(id="ov-count", className="filter-count"),
                    html.Span(id="ov-selected", className="filter-selected"),
                ],
            ),
        ],
    )

    return html.Div([
        page_header(
            "Model overview",
            "Ridge regression baseline evaluated on the most recent 20% of your Letterboxd ratings.",
        ),
        filters,
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
            className="graph-grid-2",
            children=[
                html.Div([
                    section_title("Largest under-predictions"),
                    card([data_table("ov-under", MISS_COLUMNS)]),
                ]),
                html.Div([
                    section_title("Largest over-predictions"),
                    card([data_table("ov-over", MISS_COLUMNS)]),
                ]),
            ],
        ),
    ])


def _miss_records(df: pd.DataFrame, n: int = 10) -> tuple[list[dict], list[dict]]:
    """Top-n under- and over-prediction rows formatted for the tables."""
    display = df[["Date", "Name", "Year", "Rating", "pred", "err"]].copy()
    display["Date"] = display["Date"].dt.date.astype(str)
    display["pred"] = display["pred"].round(2)
    display["err"] = display["err"].round(2)

    under = display.sort_values("err").head(n).to_dict("records")
    over = display.sort_values("err", ascending=False).head(n).to_dict("records")
    return under, over


def _metric_cards(df: pd.DataFrame) -> html.Div:
    """Metric card row for the filtered holdout set (placeholders if too small)."""
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
    Output("ov-under", "data"),
    Output("ov-over", "data"),
    Output("ov-under", "style_data_conditional"),
    Output("ov-over", "style_data_conditional"),
    Output("ov-count", "children"),
    Output("ov-selected", "children"),
    Input("ov-genre", "value"),
    Input("ov-lang", "value"),
    Input("ov-years", "value"),
    Input("ov-rating", "value"),
    Input("ov-scatter", "clickData"),
)
def update_overview(genres, languages, years, ratings, click_data):
    """Re-render all overview charts and tables from the active filters."""
    holdout = data.holdout_films()
    df = data.filter_films(holdout, genres, languages, years, ratings)

    count_text = f"Showing {len(df)} of {len(holdout)} holdout films"

    if df.empty:
        empty = empty_figure()
        return (
            _metric_cards(df),
            empty,
            empty,
            empty,
            [],
            [],
            _highlight_styles(None),
            _highlight_styles(None),
            (count_text),
            "",
        )

    selected_name, selected_text = _selected_film(click_data, df)
    under, over = _miss_records(df)
    styles = _highlight_styles(selected_name)

    return (
        _metric_cards(df),
        predicted_vs_actual_figure(df),
        error_histogram_figure(df),
        mae_by_rating_figure(df),
        under,
        over,
        styles,
        styles,
        count_text,
        selected_text,
    )
