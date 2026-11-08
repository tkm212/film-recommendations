"""Films explorer page: searchable table with posters and a detail panel."""

import dash
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from dash.development.base_component import Component

from .. import data
from ..components import card, data_table, page_header
from ..film_detail import detail_placeholder, film_detail_card, resolve_film_row

TABLE_COLUMNS = [
    {"name": "", "id": "poster", "presentation": "markdown"},
    {"name": "Film", "id": "Name"},
    {"name": "Year", "id": "Year"},
    {"name": "Genres", "id": "genres_display"},
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
            html.Div(className="toolbar-meta", children=[html.Span(id="fx-count", className="filter-count")]),
        ],
    )

    return html.Div([
        toolbar,
        page_header(
            "Film explorer",
            f"All {len(films)} rated films with out-of-fold predictions. Click a row to see details.",
        ),
        html.Div(
            className="explorer-grid",
            children=[
                card([data_table("fx-table", TABLE_COLUMNS, page_size=10, sortable=True)]),
                html.Div(id="fx-detail", children=detail_placeholder("Select a film in the table to see its details.")),
            ],
        ),
    ])


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
)
def update_table(search, genres):
    """Filter the explorer table by search text and genre."""
    df = data.filter_films(data.get_films(), genres=genres)
    if search:
        needle = search.strip().lower()
        mask = df["Name"].str.lower().str.contains(needle, regex=False)
        mask |= df["people_text"].str.lower().str.contains(needle, regex=False)
        df = df[mask]

    df = df.sort_values("Date", ascending=False)
    count_text = f"Showing {len(df)} of {len(data.get_films())} films"
    return _table_records(df), None, count_text


@callback(
    Output("fx-detail", "children"),
    Input("fx-table", "active_cell"),
    State("fx-table", "derived_viewport_data"),
)
def update_detail(active_cell, viewport_rows):
    """Show the detail panel for the clicked table row."""
    if not active_cell or not viewport_rows:
        return detail_placeholder("Select a film in the table to see its details.")

    row_idx = active_cell.get("row")
    if row_idx is None or row_idx >= len(viewport_rows):
        return detail_placeholder("Select a film in the table to see its details.")

    row = viewport_rows[row_idx]
    film = resolve_film_row(data.get_films(), row["Name"], row["Year"], row.get("tmdb_id"))
    if film is None:
        return detail_placeholder("Select a film in the table to see its details.")

    return film_detail_card(film)


# Registered after layout is defined so dash.page_registry picks it up.
dash.register_page(__name__, path="/films", name="Films", title="Film Recommendations — Films", layout=layout)
