"""Reusable UI components shared by all dashboard pages."""

from dash import dash_table, dcc, html

from .theme import COLORS

NAV_LINKS = [("Overview", "/"), ("Films", "/films"), ("About", "/about")]


def navbar() -> html.Header:
    """Top app bar with brand and page navigation."""
    return html.Header(
        className="navbar",
        children=[
            dcc.Link(
                className="nav-brand",
                href="/",
                children=[
                    html.Span(className="nav-logo", children="FR"),
                    html.Span("Film Recommendations", className="nav-title"),
                ],
            ),
            html.Nav(
                className="nav-links",
                children=[dcc.Link(name, href=path, className="nav-link") for name, path in NAV_LINKS],
            ),
        ],
    )


def card(children: list, **style_overrides: str) -> html.Div:
    """White elevated card container.

    Args:
        children (list): Card contents.
        **style_overrides (str): Extra CSS properties merged into the style.
    """
    return html.Div(children=children, className="card", style=style_overrides or None)


def page_header(title: str, subtitle: str = "") -> html.Div:
    """Page title with an optional muted subtitle."""
    children: list = [html.H1(title, className="page-title")]
    if subtitle:
        children.append(html.P(subtitle, className="page-subtitle"))
    return html.Div(children=children, className="page-header")


def section_title(text: str) -> html.H2:
    """Section heading used between cards."""
    return html.H2(text, className="section-title")


def metric_card(label: str, value: str, caption: str = "") -> html.Div:
    """Single metric card with label, large value, and optional caption."""
    children: list = [
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value"),
    ]
    if caption:
        children.append(html.Div(caption, className="metric-caption"))
    return html.Div(children=children, className="card metric-card")


def metric_row(cards: list[html.Div]) -> html.Div:
    """Wrapping horizontal row of metric cards."""
    return html.Div(children=cards, className="metric-row")


def control(label: str, component: object) -> html.Div:
    """A filter control with a small caption label above it."""
    return html.Div(children=[html.Label(label, className="control-label"), component], className="control")


def chip(text: str, kind: str = "default") -> html.Span:
    """Small rounded chip (kind: default, primary, success, danger)."""
    return html.Span(text, className=f"chip chip-{kind}")


BASE_TABLE_STYLE = {
    "style_table": {"overflowX": "auto"},
    "style_header": {
        "backgroundColor": COLORS["surface"],
        "color": COLORS["text_secondary"],
        "fontWeight": "600",
        "fontSize": "11px",
        "textTransform": "uppercase",
        "letterSpacing": "0.6px",
        "borderBottom": f"1px solid {COLORS['border']}",
        "padding": "10px 12px",
    },
    "style_cell": {
        "backgroundColor": COLORS["surface"],
        "color": COLORS["text"],
        "padding": "10px 12px",
        "fontSize": "13px",
        "textAlign": "left",
        "border": "none",
        "borderBottom": f"1px solid {COLORS['divider']}",
        "maxWidth": "320px",
        "overflow": "hidden",
        "textOverflow": "ellipsis",
    },
}

ERR_COLOR_STYLES = [
    {"if": {"filter_query": "{err} > 0.5", "column_id": "err"}, "color": COLORS["danger"], "fontWeight": "600"},
    {"if": {"filter_query": "{err} < -0.5", "column_id": "err"}, "color": COLORS["success"], "fontWeight": "600"},
]


def data_table(table_id: str, columns: list[dict], page_size: int | None = None, sortable: bool = False):
    """Themed DataTable shell; rows are supplied by callbacks via the data prop.

    Args:
        table_id (str): Component id.
        columns (list[dict]): Dash DataTable column definitions.
        page_size (int | None): Rows per page (enables paging when set).
        sortable (bool): Enable native column sorting.

    Returns:
        dash_table.DataTable: The configured table.
    """
    kwargs: dict = dict(BASE_TABLE_STYLE)
    if page_size:
        kwargs["page_size"] = page_size
    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=[],
        sort_action="native" if sortable else "none",
        style_data_conditional=ERR_COLOR_STYLES,
        cell_selectable=True,
        markdown_options={"link_target": "_blank"},
        **kwargs,
    )
