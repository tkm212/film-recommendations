"""Reusable Dash building blocks for model-results dashboards.

Compose dashboards from three layers:

- **Theme + components**: `page`, `heading`, `metric_card`, `metric_row`,
  `styled_table` — styled primitives that keep new dashboards consistent.
- **Figures**: `predicted_vs_actual_figure`, `error_histogram_figure`,
  `mae_by_rating_figure` — plotly figures built from a predictions DataFrame.
- **Assembly**: `compute_metrics` and `build_results_app` — the default
  results dashboard, itself just a composition of the pieces above.

A predictions ("eval") DataFrame is expected to have columns Date, Name,
Year, Rating (true), pred (clipped prediction) and err (pred - Rating).

Example of a custom dashboard:

    from dash import Dash, dcc
    from film_recommendations import dashboard as dbd

    app = Dash(__name__, title="My dashboard")
    app.layout = dbd.page([
        dbd.heading("My custom view", level=1),
        dbd.metric_row([dbd.metric_card("MAE", "0.498")]),
        dcc.Graph(figure=dbd.error_histogram_figure(eval_df)),
        dbd.styled_table("misses", eval_df.head(10)),
    ])
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .constants import DEFAULT_SNAP_STEP, RATING_CLIP_MAX, RATING_CLIP_MIN

# --- Theme ---

THEME = {
    "background": "#0e1319",
    "surface": "#1e2530",
    "surface_alt": "#141a22",
    "text": "#f0f3f7",
    "text_body": "#dfe5ec",
    "muted": "#9aa5b1",
    "font": "system-ui, sans-serif",
}
PLOT_TEMPLATE = "plotly_dark"

CARD_STYLE = {
    "flex": "1",
    "padding": "16px 20px",
    "borderRadius": "10px",
    "backgroundColor": THEME["surface"],
    "color": THEME["text"],
    "textAlign": "center",
    "minWidth": "140px",
}


# --- Components ---


def page(children: list, **style_overrides: str) -> html.Div:
    """Wrap dashboard content in a full-page themed container.

    Args:
        children (list): Dash components to render on the page.
        **style_overrides (str): Extra CSS properties merged into the page style.

    Returns:
        html.Div: The page container.
    """
    style = {
        "backgroundColor": THEME["background"],
        "minHeight": "100vh",
        "padding": "24px 32px",
        "fontFamily": THEME["font"],
        **style_overrides,
    }
    return html.Div(children=children, style=style)


def heading(text: str, level: int = 2, **style_overrides: str) -> html.H1 | html.H2 | html.P:
    """Create a themed heading (level 1 or 2) or muted subtitle (level 3+).

    Args:
        text (str): Heading text.
        level (int): 1 for page title, 2 for section, 3+ for muted subtitle.
        **style_overrides (str): Extra CSS properties merged into the style.

    Returns:
        html.H1 | html.H2 | html.P: The heading component.
    """
    if level == 1:
        style = {"color": THEME["text"], "fontSize": "24px", "marginBottom": "4px", **style_overrides}
        return html.H1(text, style=style)
    if level == 2:
        style = {"color": THEME["text"], "fontSize": "18px", **style_overrides}
        return html.H2(text, style=style)
    style = {"color": THEME["muted"], "marginTop": "0", **style_overrides}
    return html.P(text, style=style)


def metric_card(label: str, value: str) -> html.Div:
    """Create a single metric card with a label and a large value.

    Args:
        label (str): Short metric name shown above the value.
        value (str): Pre-formatted metric value.

    Returns:
        html.Div: The metric card.
    """
    return html.Div(
        [
            html.Div(label, style={"fontSize": "13px", "color": THEME["muted"], "marginBottom": "6px"}),
            html.Div(value, style={"fontSize": "26px", "fontWeight": "600"}),
        ],
        style=CARD_STYLE,
    )


def metric_row(cards: list[html.Div]) -> html.Div:
    """Lay out metric cards in a wrapping horizontal row.

    Args:
        cards (list[html.Div]): Cards from metric_card.

    Returns:
        html.Div: The row container.
    """
    return html.Div(
        children=cards,
        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "24px"},
    )


def graph_grid(figures: list[go.Figure], columns: int = 2) -> html.Div:
    """Lay out plotly figures in a responsive grid.

    Args:
        figures (list[go.Figure]): Figures to render.
        columns (int): Number of grid columns. Defaults to 2.

    Returns:
        html.Div: The grid container.
    """
    return html.Div(
        children=[dcc.Graph(figure=fig) for fig in figures],
        style={"display": "grid", "gridTemplateColumns": " ".join(["1fr"] * columns), "gap": "16px"},
    )


def styled_table(table_id: str, df: pd.DataFrame, round_decimals: int = 2) -> dash_table.DataTable:
    """Render a DataFrame as a themed Dash DataTable.

    Args:
        table_id (str): Unique component id.
        df (pd.DataFrame): Data to display; float columns are rounded and
            datetime columns shown as dates.
        round_decimals (int): Decimal places for float columns. Defaults to 2.

    Returns:
        dash_table.DataTable: The styled table.
    """
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_datetime64_any_dtype(display[col]):
            display[col] = display[col].dt.date.astype(str)
        elif pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].round(round_decimals)

    return dash_table.DataTable(
        id=table_id,
        data=display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": THEME["surface"], "color": THEME["text"], "fontWeight": "600"},
        style_cell={
            "backgroundColor": THEME["surface_alt"],
            "color": THEME["text_body"],
            "padding": "8px",
            "fontFamily": THEME["font"],
            "fontSize": "13px",
            "textAlign": "left",
        },
    )


# --- Metrics and figures ---


def compute_metrics(eval_df: pd.DataFrame) -> dict[str, float]:
    """Compute headline evaluation metrics from a predictions DataFrame.

    Args:
        eval_df (pd.DataFrame): DataFrame with Rating (true) and pred
            (clipped prediction) columns.

    Returns:
        dict[str, float]: mae, rmse, r2, spearman, exact_match_rate and
        within_half_star.
    """
    y_true = eval_df["Rating"]
    y_pred = eval_df["pred"]
    pred_snapped = np.round(y_pred / DEFAULT_SNAP_STEP) * DEFAULT_SNAP_STEP

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred).correlation,
        "exact_match_rate": (pred_snapped == y_true).mean(),
        "within_half_star": (np.abs(pred_snapped - y_true) <= DEFAULT_SNAP_STEP).mean(),
    }


def predicted_vs_actual_figure(eval_df: pd.DataFrame) -> go.Figure:
    """Scatter of predicted vs actual ratings with an identity reference line.

    Args:
        eval_df (pd.DataFrame): Predictions DataFrame (Rating, pred, Name,
            Year, Date).

    Returns:
        go.Figure: The scatter figure.
    """
    fig = px.scatter(
        eval_df,
        x="Rating",
        y="pred",
        hover_data=["Name", "Year", "Date"],
        labels={"Rating": "Actual rating", "pred": "Predicted rating"},
        title="Predicted vs actual ratings (holdout set)",
        template=PLOT_TEMPLATE,
    )
    fig.add_shape(
        type="line",
        x0=RATING_CLIP_MIN,
        y0=RATING_CLIP_MIN,
        x1=RATING_CLIP_MAX,
        y1=RATING_CLIP_MAX,
        line={"dash": "dash", "color": THEME["muted"]},
    )
    return fig


def error_histogram_figure(eval_df: pd.DataFrame, nbins: int = 25) -> go.Figure:
    """Histogram of prediction errors.

    Args:
        eval_df (pd.DataFrame): Predictions DataFrame with an err column.
        nbins (int): Number of histogram bins. Defaults to 25.

    Returns:
        go.Figure: The histogram figure.
    """
    return px.histogram(
        eval_df,
        x="err",
        nbins=nbins,
        labels={"err": "Prediction error (pred - actual)"},
        title="Error distribution",
        template=PLOT_TEMPLATE,
    )


def mae_by_rating_figure(eval_df: pd.DataFrame) -> go.Figure:
    """Bar chart of mean absolute error grouped by true rating.

    Args:
        eval_df (pd.DataFrame): Predictions DataFrame with Rating and err.

    Returns:
        go.Figure: The bar figure.
    """
    bucket_mae = eval_df.assign(abs_err=lambda d: d["err"].abs()).groupby("Rating")["abs_err"].mean().reset_index()
    return px.bar(
        bucket_mae,
        x="Rating",
        y="abs_err",
        labels={"Rating": "True rating", "abs_err": "Mean absolute error"},
        title="MAE by true rating",
        template=PLOT_TEMPLATE,
    )


# --- Assembly ---

MISS_TABLE_COLUMNS = ["Date", "Name", "Year", "Rating", "pred", "err"]


def build_results_app(
    eval_df: pd.DataFrame,
    title: str = "Letterboxd rating model — Ridge baseline",
    n_misses: int = 10,
) -> Dash:
    """Build the default model-results dashboard.

    Args:
        eval_df (pd.DataFrame): Holdout predictions with columns Date, Name,
            Year, Rating, pred, err.
        title (str): Page title. Defaults to the Ridge baseline title.
        n_misses (int): Rows shown in each largest-misses table. Defaults to 10.

    Returns:
        Dash: The configured Dash application.
    """
    metrics = compute_metrics(eval_df)
    under = eval_df.sort_values("err").head(n_misses)[MISS_TABLE_COLUMNS]
    over = eval_df.sort_values("err").tail(n_misses).iloc[::-1][MISS_TABLE_COLUMNS]

    app = Dash(__name__, title=title)
    app.layout = page([
        heading(title, level=1),
        heading(f"Holdout set: {len(eval_df)} most recent ratings", level=3),
        metric_row([
            metric_card("MAE", f"{metrics['mae']:.3f}"),
            metric_card("RMSE", f"{metrics['rmse']:.3f}"),
            metric_card("R²", f"{metrics['r2']:.3f}"),
            metric_card("Spearman rho", f"{metrics['spearman']:.3f}"),
            metric_card("Exact match (0.5★)", f"{metrics['exact_match_rate']:.1%}"),
            metric_card("Within 0.5★", f"{metrics['within_half_star']:.1%}"),
        ]),
        graph_grid([predicted_vs_actual_figure(eval_df), error_histogram_figure(eval_df)]),
        graph_grid([mae_by_rating_figure(eval_df)], columns=1),
        heading("Largest under-predictions"),
        styled_table("under-table", under),
        heading("Largest over-predictions", marginTop="24px"),
        styled_table("over-table", over),
    ])
    return app
