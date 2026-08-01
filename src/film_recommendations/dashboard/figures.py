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


def model_comparison_figure(comparison_df: pd.DataFrame) -> go.Figure:
    """Bar chart of out-of-fold MAE by model variant (best model highlighted)."""
    plot_df = comparison_df.sort_values("mae", ascending=True)
    colors = [COLORS["primary"] if row["is_best"] else COLORS["text_secondary"] for _, row in plot_df.iterrows()]
    fig = go.Figure(
        go.Bar(
            x=plot_df["mae"],
            y=plot_df["model"],
            orientation="h",
            marker={"color": colors, "cornerradius": 4},
            hovertemplate="Model: %{y}<br>MAE: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=max(FIGURE_HEIGHT, 72 * len(plot_df)),
        title="Model comparison (3-fold out-of-fold MAE)",
        xaxis_title="Mean absolute error (stars)",
        yaxis_title="",
    )
    return fig


def group_coefficient_figure(groups_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of total |coefficient| mass by feature group."""
    plot_df = groups_df.sort_values("total_abs_coefficient", ascending=True)
    fig = px.bar(
        plot_df,
        x="total_abs_coefficient",
        y="group_label",
        orientation="h",
        labels={"total_abs_coefficient": "Sum of |coefficients|", "group_label": "Feature group"},
        title="Ridge coefficient mass by feature group",
        template=PLOT_TEMPLATE,
        height=FIGURE_HEIGHT,
    )
    fig.update_traces(marker={"cornerradius": 4})
    return fig


def group_permutation_figure(perm_df: pd.DataFrame) -> go.Figure:
    """Bar chart of MAE increase when shuffling each feature group."""
    plot_df = perm_df.sort_values("importance_mean", ascending=True)
    fig = px.bar(
        plot_df,
        x="importance_mean",
        y="group_label",
        orientation="h",
        error_x="importance_std",
        labels={"importance_mean": "MAE increase when group shuffled", "group_label": "Feature group"},
        title="Permutation importance by feature group",
        template=PLOT_TEMPLATE,
        height=FIGURE_HEIGHT,
    )
    fig.update_traces(marker={"cornerradius": 4})
    return fig


def top_coefficients_figure(coef_df: pd.DataFrame, n: int = 20, group: str | None = None) -> go.Figure:
    """Horizontal bar chart of the largest signed Ridge coefficients."""
    plot_df = coef_df if group is None else coef_df[coef_df["group"] == group]
    plot_df = plot_df.nlargest(n, "abs_coefficient").sort_values("coefficient", ascending=True)
    if plot_df.empty:
        return empty_figure("No features in this group")
    colors = [COLORS["success"] if c > 0 else COLORS["danger"] for c in plot_df["coefficient"]]

    fig = go.Figure(
        go.Bar(
            x=plot_df["coefficient"],
            y=plot_df["feature"],
            orientation="h",
            marker={"color": colors, "cornerradius": 4},
            hovertemplate="Feature: %{y}<br>Coefficient: %{x:.4f}<extra></extra>",
        )
    )
    title = "Top Ridge coefficients"
    if group:
        title += f" ({plot_df['group_label'].iloc[0] if len(plot_df) else group})"
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=max(FIGURE_HEIGHT, 28 * len(plot_df)),
        title=title,
        xaxis_title="Effect on predicted rating (stars)",
        yaxis_title="",
    )
    return fig
