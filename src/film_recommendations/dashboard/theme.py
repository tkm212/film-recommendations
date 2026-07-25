"""Design tokens and the plotly template used across all dashboard pages."""

import plotly.graph_objects as go
import plotly.io as pio

COLORS = {
    "background": "#f8f9fa",
    "surface": "#ffffff",
    "border": "#dadce0",
    "divider": "#f1f3f4",
    "text": "#202124",
    "text_secondary": "#5f6368",
    "primary": "#1a73e8",
    "primary_soft": "#e8f0fe",
    "success": "#188038",
    "success_soft": "#e6f4ea",
    "danger": "#d93025",
    "danger_soft": "#fce8e6",
    "warning": "#f9ab00",
}

FONT_FAMILY = "'Inter', 'Roboto', 'Segoe UI', system-ui, sans-serif"
FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"

CHART_COLORWAY = ["#1a73e8", "#188038", "#d93025", "#f9ab00", "#9334e6", "#007b83"]

PLOT_TEMPLATE = "film_light"

_axis = {
    "gridcolor": COLORS["divider"],
    "zerolinecolor": "#e8eaed",
    "linecolor": "#e8eaed",
    "ticks": "",
    "title": {"font": {"size": 12}},
}

pio.templates[PLOT_TEMPLATE] = go.layout.Template(
    layout={
        "font": {"family": FONT_FAMILY, "color": COLORS["text_secondary"], "size": 12},
        "title": {"font": {"family": FONT_FAMILY, "color": COLORS["text"], "size": 15}, "x": 0.02, "xanchor": "left"},
        "paper_bgcolor": COLORS["surface"],
        "plot_bgcolor": COLORS["surface"],
        "colorway": CHART_COLORWAY,
        "xaxis": _axis,
        "yaxis": _axis,
        "margin": {"t": 56, "r": 16, "b": 48, "l": 56},
        "hoverlabel": {"font": {"family": FONT_FAMILY, "size": 12}},
        "legend": {"font": {"size": 12}},
    }
)
