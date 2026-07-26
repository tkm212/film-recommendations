"""Plotly figure builders for the dashboard pages."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..constants import RATING_CLIP_MAX, RATING_CLIP_MIN
from .theme import COLORS, PLOT_TEMPLATE

FIGURE_HEIGHT = 380


def empty_figure(message: str = "No films match the current filters") -> go.Figure:
    """Placeholder figure shown when a filter selection returns no data."""
    fig = go.Figure()
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=FIGURE_HEIGHT,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "showarrow": False,
                "font": {"size": 14, "color": COLORS["text_secondary"]},
            }
        ],
    )
    return fig


def predicted_vs_actual_figure(eval_df: pd.DataFrame) -> go.Figure:
    """Scatter of predicted vs actual ratings with an identity reference line.

    Args:
        eval_df (pd.DataFrame): Predictions DataFrame (Rating, pred, Name,
            Year, Date). Points carry the film Name as customdata so click
            events can identify them.
    """
    fig = px.scatter(
        eval_df,
        x="Rating",
        y="pred",
        custom_data=["Name"],
        hover_data=["Name", "Year", "Date"],
        labels={"Rating": "Actual rating", "pred": "Predicted rating"},
        title="Predicted vs actual",
        template=PLOT_TEMPLATE,
        height=FIGURE_HEIGHT,
    )
    fig.update_traces(marker={"size": 9, "opacity": 0.75, "line": {"width": 1, "color": "#ffffff"}})
    fig.add_shape(
        type="line",
        x0=RATING_CLIP_MIN,
        y0=RATING_CLIP_MIN,
        x1=RATING_CLIP_MAX,
        y1=RATING_CLIP_MAX,
        line={"dash": "dash", "color": COLORS["border"]},
    )
    return fig


def error_histogram_figure(eval_df: pd.DataFrame, nbins: int = 25) -> go.Figure:
    """Histogram of prediction errors.

    Args:
        eval_df (pd.DataFrame): Predictions DataFrame with an err column.
        nbins (int): Number of histogram bins. Defaults to 25.
    """
    fig = px.histogram(
        eval_df,
        x="err",
        nbins=nbins,
        labels={"err": "Prediction error (predicted - actual)"},
        title="Error distribution",
        template=PLOT_TEMPLATE,
        height=FIGURE_HEIGHT,
    )
    fig.update_traces(marker={"line": {"width": 1, "color": "#ffffff"}})
    return fig


def mae_by_rating_figure(eval_df: pd.DataFrame) -> go.Figure:
    """Bar chart of mean absolute error grouped by true rating.

    Args:
        eval_df (pd.DataFrame): Predictions DataFrame with Rating and err.
    """
    bucket_mae = eval_df.assign(abs_err=lambda d: d["err"].abs()).groupby("Rating")["abs_err"].mean().reset_index()
    fig = px.bar(
        bucket_mae,
        x="Rating",
        y="abs_err",
        labels={"Rating": "Actual rating", "abs_err": "Mean absolute error"},
        title="Where the model misses: MAE by actual rating",
        template=PLOT_TEMPLATE,
        height=FIGURE_HEIGHT,
    )
    fig.update_traces(marker={"cornerradius": 4})
    return fig
