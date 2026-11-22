"""Discovery recommendations page with stochastic refreshes."""

from __future__ import annotations

import dash
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, callback_context, dcc, html

from ...discovery import discover_recommendations
from .. import data
from ..components import card, data_table, page_header
from ..theme import COLORS

TABLE_COLUMNS = [
    {"name": "Film", "id": "Name"},
    {"name": "Year", "id": "Year"},
    {"name": "Match", "id": "similarity", "type": "numeric", "format": {"specifier": ".3f"}},
    {"name": "Predicted", "id": "pred"},
    {"name": "Genres", "id": "genres"},
    {"name": "Because you liked", "id": "anchor"},
]


def _format_rows(recs: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for _, row in recs.iterrows():
        pred = pd.to_numeric(row.get("pred"), errors="coerce")
        genres = str(row.get("genres", "") or "").replace("|", ", ")
        anchor = row.get("anchor_film")
        anchor_year = row.get("anchor_year")
        because = f"{anchor} ({anchor_year})" if pd.notna(anchor_year) and anchor else "—"
        rows.append({
            "Name": row["Name"],
            "Year": row["Year"],
            "similarity": row.get("similarity"),
            "pred": f"{pred:.2f}" if pd.notna(pred) else "—",
            "genres": genres[:80] + ("…" if len(genres) > 80 else ""),
            "anchor": because,
        })
    return rows


def layout() -> html.Div:
    if not data.has_taste_space() or data.get_taste_embeddings() is None:
        return html.Div([
            page_header("Discover", "Films to watch next, picked from TMDB space."),
            card([
                html.P(
                    "Run ingest and train so the taste-map universe and embeddings exist, "
                    "then reload the dashboard.",
                    style={"color": COLORS["text_secondary"], "margin": 0},
                ),
            ]),
        ])

    toolbar = html.Div(
        className="toolbar",
        children=[
            html.Div(
                className="toolbar-item",
                children=[
                    html.Button("Shuffle picks", id="rec-refresh", n_clicks=0, className="nav-link"),
                ],
            ),
            html.Div(
                className="toolbar-item toolbar-slider",
                children=[
                    html.Span("How many", className="toolbar-label"),
                    dcc.Slider(id="rec-n", min=8, max=30, step=1, value=18, marks={8: "8", 18: "18", 30: "30"}),
                ],
            ),
            html.Div(
                className="toolbar-item toolbar-slider",
                children=[
                    html.Span("Surprise", className="toolbar-label"),
                    dcc.Slider(
                        id="rec-temp",
                        min=0.0,
                        max=0.35,
                        step=0.05,
                        value=0.12,
                        marks={0: "stable", 0.12: "mixed", 0.35: "wild"},
                    ),
                ],
            ),
            html.Div(className="toolbar-meta", children=[html.Span(id="rec-hint", className="filter-count")]),
        ],
    )

    return html.Div([
        page_header(
            "Discover",
            "Embedding-space picks outside your library. Random liked-film anchors and Gumbel noise "
            "keep each shuffle feeling fresh.",
        ),
        toolbar,
        dcc.Store(id="rec-seed"),
        card([data_table("rec-table", TABLE_COLUMNS, page_size=12, sortable=True)]),
    ])


@callback(
    Output("rec-seed", "data"),
    Output("rec-table", "data"),
    Output("rec-hint", "children"),
    Input("rec-refresh", "n_clicks"),
    Input("rec-n", "value"),
    Input("rec-temp", "value"),
    State("rec-seed", "data"),
)
def refresh_recommendations(n_clicks, n_films, temperature, stored_seed):
    if not data.has_taste_space():
        return stored_seed, [], ""

    embeddings = data.get_taste_embeddings()
    if embeddings is None:
        return stored_seed, [], ""

    triggered = callback_context.triggered_id
    seed = stored_seed
    if triggered == "rec-refresh" or seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))

    universe = data.get_taste_films()
    library = data.library_films()
    exclude = data.library_film_keys()

    n_films = int(n_films or 18)
    temperature = float(temperature if temperature is not None else 0.12)

    recs = discover_recommendations(
        universe,
        embeddings,
        library,
        exclude,
        n=n_films,
        temperature=temperature,
        seed=int(seed),
    )

    hint = f"{len(recs)} picks · seed {seed} · shuffle for new anchors"
    return seed, _format_rows(recs), hint


dash.register_page(__name__, path="/discover", name="Discover", title="Film Recommendations — Discover", layout=layout)