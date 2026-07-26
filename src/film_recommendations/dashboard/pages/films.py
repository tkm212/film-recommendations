"""Films explorer page: searchable table with posters and a detail panel."""

import dash
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from dash.development.base_component import Component

from .. import data
from ..components import card, chip, data_table, page_header
from ..theme import COLORS

TABLE_COLUMNS = [
    {"name": "", "id": "poster", "presentation": "markdown"},
    {"name": "Film", "id": "Name"},
    {"name": "Year", "id": "Year"},
    {"name": "Genres", "id": "genres_display"},
    {"name": "Your rating", "id": "Rating"},
    {"name": "Predicted", "id": "pred"},
    {"name": "Error", "id": "err"},
]

SPLIT_OPTIONS = [
    {"label": "All films", "value": "all"},
    {"label": "Holdout only", "value": "test"},
    {"label": "Training only", "value": "train"},
]


def _toolbar_item(label: str, component: Component, extra_class: str = "") -> html.Div:
    return html.Div(
        className=f"toolbar-item {extra_class}".strip(),
        children=[html.Span(label, className="toolbar-label"), component],
    )


def layout() -> html.Div:
    """Build the films explorer layout."""
    films = data.get_films()

    toolbar = html.Div(
        className="toolbar",
        children=[
            _toolbar_item(
                "Search",
                dcc.Input(
                    id="fx-search",
                    type="text",
                    placeholder="Films, directors, actors…",
                    debounce=True,
                    className="search-input",
                ),
                "toolbar-search",
            ),
            _toolbar_item(
                "Genre",
                dcc.Dropdown(id="fx-genre", options=data.genre_options(), multi=True, placeholder="All"),
                "toolbar-dd",
            ),
            _toolbar_item(
                "Data split",
                dcc.Dropdown(id="fx-split", options=SPLIT_OPTIONS, value="all", clearable=False),
                "toolbar-dd",
            ),
            html.Div(className="toolbar-meta", children=[html.Span(id="fx-count", className="filter-count")]),
        ],
    )

    return html.Div([
        toolbar,
        page_header(
            "Film explorer",
            f"All {len(films)} rated films with the model's predictions. Click a row to see details.",
        ),
        html.Div(
            className="explorer-grid",
            children=[
                card([data_table("fx-table", TABLE_COLUMNS, page_size=10, sortable=True)]),
                html.Div(id="fx-detail", children=_detail_placeholder()),
            ],
        ),
    ])


def _detail_placeholder() -> html.Div:
    return card(
        [html.Div("Select a film in the table to see its details.", className="detail-placeholder")],
    )


def _table_records(df: pd.DataFrame) -> list[dict]:
    """Format films rows for the explorer table (tmdb_id kept for lookups)."""
    records = []
    for _, row in df.iterrows():
        url = data.poster_url(row["poster_path"], "w92")
        records.append({
            "poster": f"![poster]({url})" if url else "",
            "Name": row["Name"],
            "Year": row["Year"],
            "genres_display": row["genres"].replace("|", ", "),
            "Rating": row["Rating"],
            "pred": round(row["pred"], 2),
            "err": round(row["err"], 2),
            "tmdb_id": row["tmdb_id"],
        })
    return records


@callback(
    Output("fx-table", "data"),
    Output("fx-table", "active_cell"),
    Output("fx-count", "children"),
    Input("fx-search", "value"),
    Input("fx-genre", "value"),
    Input("fx-split", "value"),
)
def update_table(search, genres, split):
    """Filter the explorer table by search text, genre, and data split."""
    df = data.get_films()

    if split and split != "all":
        df = df[df["split"] == split]
    df = data.filter_films(df, genres=genres)
    if search:
        needle = search.strip().lower()
        mask = df["Name"].str.lower().str.contains(needle, regex=False)
        mask |= df["people_text"].str.lower().str.contains(needle, regex=False)
        df = df[mask]

    df = df.sort_values("Date", ascending=False)
    count_text = f"Showing {len(df)} of {len(data.get_films())} films"
    return _table_records(df), None, count_text


def _rating_block(label: str, value: str) -> html.Div:
    return html.Div(
        className="detail-rating",
        children=[html.Div(label, className="metric-label"), html.Div(value, className="detail-rating-value")],
    )


def _detail_card(row: pd.Series) -> html.Div:
    """Full detail card for one film row."""
    poster = data.poster_url(row["poster_path"], "w342")
    err = row["err"]
    err_color = COLORS["danger"] if err > 0 else COLORS["success"]
    verdict = "model over-predicted" if err > 0 else "model under-predicted"
    genres = [g for g in row["genres"].split("|") if g]

    meta_bits = []
    if pd.notna(row.get("runtime")) and row["runtime"]:
        meta_bits.append(f"{int(row['runtime'])} min")
    if row.get("original_language"):
        meta_bits.append(data.language_name(row["original_language"]))
    meta_bits.append(f"Watched {row['Date'].date()}")

    children: list = []
    if poster:
        children.append(html.Img(src=poster, className="detail-poster", alt=f"{row['Name']} poster"))
    children.extend([
        html.H3(f"{row['Name']} ({row['Year']})", className="detail-title"),
        html.Div(className="detail-chips", children=[chip(g) for g in genres] + [chip(row["split"], "primary")]),
        html.Div(" · ".join(meta_bits), className="detail-meta"),
        html.Div(
            className="detail-ratings",
            children=[
                _rating_block("Your rating", f"{row['Rating']:.1f}"),
                _rating_block("Predicted", f"{row['pred']:.2f}"),
            ],
        ),
        html.Div(f"{err:+.2f} — {verdict}", className="detail-err", style={"color": err_color}),
    ])

    if row.get("overview"):
        children.append(html.P(row["overview"], className="detail-overview"))
    if row.get("people_text"):
        children.extend([
            html.Div("Director & cast", className="metric-label"),
            html.P(row["people_text"], className="detail-people"),
        ])

    uri = row.get("Letterboxd URI", "")
    if isinstance(uri, str) and uri.startswith("http"):
        children.append(html.A("View on Letterboxd", href=uri, target="_blank", className="detail-link"))

    return card(children)


@callback(
    Output("fx-detail", "children"),
    Input("fx-table", "active_cell"),
    State("fx-table", "derived_viewport_data"),
)
def update_detail(active_cell, viewport_rows):
    """Show the detail panel for the clicked table row."""
    if not active_cell or not viewport_rows:
        return _detail_placeholder()

    row_idx = active_cell.get("row")
    if row_idx is None or row_idx >= len(viewport_rows):
        return _detail_placeholder()

    tmdb_id = viewport_rows[row_idx].get("tmdb_id")
    films = data.get_films()
    matches = films[films["tmdb_id"] == tmdb_id]
    if matches.empty:
        return _detail_placeholder()

    return _detail_card(matches.iloc[0])


# Registered after layout is defined so dash.page_registry picks it up.
dash.register_page(__name__, path="/films", name="Films", title="Film Recommendations — Films", layout=layout)
