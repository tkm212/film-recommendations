"""Plotly figure builders for the dashboard pages."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..constants import RATING_CLIP_MAX, RATING_CLIP_MIN
from .theme import CHART_COLORWAY, COLORS, PLOT_TEMPLATE

FIGURE_HEIGHT = 380


def _taste_color_encoding(color_by: str) -> tuple[str, dict, str]:
    """Return color column, plotly color kwargs, and title fragment."""
    if color_by == "rating":
        return (
            "Rating",
            {"color_continuous_scale": "Viridis", "range_color": [RATING_CLIP_MIN, RATING_CLIP_MAX]},
            "your rating",
        )
    if color_by == "pred":
        return (
            "pred",
            {"color_continuous_scale": "Viridis", "range_color": [RATING_CLIP_MIN, RATING_CLIP_MAX]},
            "model prediction",
        )
    if color_by == "residual":
        return "err", {"color_continuous_scale": "RdBu_r", "color_continuous_midpoint": 0}, "prediction residual"
    return "cluster_label", {"color_discrete_sequence": CHART_COLORWAY}, "KMeans clusters"


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


def taste_map_figure(
    films_df: pd.DataFrame,
    projection: str = "pca",
    dimensions: int = 3,
    explained_variance: tuple[float, float, float] | None = None,
    color_by: str = "cluster",
) -> go.Figure:
    """2D or 3D scatter in PCA or UMAP space with configurable color encoding."""
    from ..film_taste_space import projection_columns

    x_col, y_col, z_col = projection_columns(projection)
    if films_df.empty or x_col not in films_df.columns:
        return empty_figure("Taste map unavailable — run training to compute embeddings")

    plot_df = films_df.copy()
    if "err" not in plot_df.columns:
        plot_df["err"] = float("nan")
    if "pred" not in plot_df.columns:
        plot_df["pred"] = float("nan")
    if "tmdb_id" not in plot_df.columns:
        plot_df["tmdb_id"] = None

    proj_label = "PCA" if projection == "pca" else "UMAP"
    labels = {
        x_col: f"{proj_label} 1",
        y_col: f"{proj_label} 2",
        z_col: f"{proj_label} 3",
        "Rating": "Your rating",
        "pred": "Predicted",
        "err": "Residual (pred - rating)",
        "cluster_label": "Cluster",
        "Name": "Film",
        "Year": "Year",
        "genres": "Genres",
    }
    if projection == "pca" and explained_variance:
        labels[x_col] = f"PC 1 ({explained_variance[0]:.0%} var.)"
        labels[y_col] = f"PC 2 ({explained_variance[1]:.0%} var.)"
        if dimensions == 3:
            labels[z_col] = f"PC 3 ({explained_variance[2]:.0%} var.)"

    color_field, color_args, title_suffix = _taste_color_encoding(color_by)
    dim_label = "2D" if dimensions == 2 else "3D"
    hover = {
        "Year": True,
        "Rating": ":.1f",
        "pred": ":.1f",
        "err": ":.2f",
        "genres": True,
        x_col: False,
        y_col: False,
        z_col: False,
        "tmdb_id": False,
    }

    if dimensions == 2:
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=color_field,
            custom_data=["Name", "Year", "tmdb_id"],
            hover_name="Name",
            hover_data=hover,
            labels=labels,
            template=PLOT_TEMPLATE,
            height=520,
            **color_args,
        )
        fig.update_traces(marker={"size": 8, "opacity": 0.85, "line": {"width": 0.5, "color": "#ffffff"}})
        fig.update_layout(
            title=f"{proj_label} {dim_label} taste map (colored by {title_suffix})",
            xaxis_title=labels[x_col],
            yaxis_title=labels[y_col],
        )
        return fig

    fig = px.scatter_3d(
        plot_df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_field,
        custom_data=["Name", "Year", "tmdb_id"],
        hover_name="Name",
        hover_data=hover,
        labels=labels,
        template=PLOT_TEMPLATE,
        height=520,
        **color_args,
    )
    fig.update_traces(marker={"size": 5, "opacity": 0.88, "line": {"width": 0.5, "color": "#ffffff"}})
    fig.update_layout(
        title=f"{proj_label} {dim_label} taste map (colored by {title_suffix})",
        scene={
            "xaxis_title": labels[x_col],
            "yaxis_title": labels[y_col],
            "zaxis_title": labels[z_col],
        },
    )
    return fig


def taste_timeline_figure(films_df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of watches per year, colored by cluster."""
    if films_df.empty or "cluster_label" not in films_df.columns:
        return empty_figure("No timeline data")
    plot_df = films_df.copy()
    plot_df["year_watched"] = pd.to_datetime(plot_df["Date"]).dt.year
    counts = plot_df.groupby(["year_watched", "cluster_label"], observed=True).size().to_frame("films").reset_index()
    fig = px.bar(
        counts,
        x="year_watched",
        y="films",
        color="cluster_label",
        labels={"year_watched": "Year watched", "films": "Films", "cluster_label": "Cluster"},
        title="Watch history by year and cluster",
        template=PLOT_TEMPLATE,
        height=FIGURE_HEIGHT,
        color_discrete_sequence=CHART_COLORWAY,
    )
    fig.update_traces(marker={"cornerradius": 2})
    return fig


def taste_map_3d_figure(
    films_df: pd.DataFrame,
    explained_variance: tuple[float, float, float] | None = None,
    color_by: str = "cluster",
) -> go.Figure:
    """Backward-compatible alias for PCA 3D taste map."""
    return taste_map_figure(
        films_df,
        projection="pca",
        dimensions=3,
        explained_variance=explained_variance,
        color_by=color_by,
    )


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
