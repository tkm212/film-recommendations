"""Taste map page: PCA/UMAP embedding space with clusters and discovery tools."""

import dash
import pandas as pd
from dash import Input, Output, callback, callback_context, dcc, html
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from ...film_taste_space import (
    cluster_insights,
    cluster_summary,
    nearest_neighbor_films,
    nonlinear_projection_label,
    recommend_similar_films,
)
from .. import data
from ..components import card, data_table, page_header, section_title
from ..figures import empty_figure, taste_map_figure, taste_reference_figure, taste_timeline_figure
from ..film_detail import detail_placeholder, film_detail_card, resolve_film_row
from ..theme import COLORS

COLOR_BY_OPTIONS = [
    {"label": "Cluster", "value": "cluster"},
    {"label": "Your rating", "value": "rating"},
    {"label": "Model prediction", "value": "pred"},
    {"label": "Residual (pred - rating)", "value": "residual"},
]

PROJECTION_OPTIONS = [
    {"label": "PCA (linear)", "value": "pca"},
    {"label": nonlinear_projection_label(), "value": "umap"},
]

DIMENSION_OPTIONS = [
    {"label": "2D", "value": 2},
    {"label": "3D", "value": 3},
]

SUMMARY_COLUMNS = [
    {"name": "Cluster", "id": "cluster_label"},
    {"name": "Films", "id": "films", "type": "numeric"},
    {"name": "Mean rating", "id": "mean_rating", "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "Mean predicted", "id": "mean_pred", "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "Top genre", "id": "top_genre"},
]

NEIGHBOR_COLUMNS = [
    {"name": "Film", "id": "Name"},
    {"name": "Year", "id": "Year"},
    {"name": "Similarity", "id": "similarity", "type": "numeric", "format": {"specifier": ".3f"}},
    {"name": "Cluster", "id": "cluster_label"},
    {"name": "Your rating", "id": "Rating"},
]

RECOMMEND_COLUMNS = [
    {"name": "Film", "id": "Name"},
    {"name": "Year", "id": "Year"},
    {"name": "Similarity", "id": "similarity", "type": "numeric", "format": {"specifier": ".3f"}},
    {"name": "Predicted", "id": "pred"},
    {"name": "Genres", "id": "genres"},
]


def _toolbar_item(label: str, component: Component, extra_class: str = "") -> html.Div:
    return html.Div(
        className=f"toolbar-item {extra_class}".strip(),
        children=[html.Span(label, className="toolbar-label"), component],
    )


def _cluster_cards(cards: list[dict]) -> html.Div:
    if not cards:
        return html.Div("No clusters in the current filter.", className="detail-placeholder")
    items = []
    for info in cards:
        posters = []
        for _, row in info["sample"].iterrows():
            url = data.poster_url(str(row.get("poster_path", "") or ""), "w92")
            if url:
                posters.append(html.Img(src=url, className="cluster-poster-thumb", alt=row["Name"]))
        kw = ", ".join(info["keywords"][:5]) or "—"
        people = ", ".join(info["people"][:5]) or "—"
        items.append(
            html.Div(
                className="cluster-card",
                children=[
                    html.Div(info["cluster_label"], className="cluster-card-title"),
                    html.Div(
                        f"{info['films']} films · lead genre: {info['top_genre']}",
                        className="cluster-card-meta",
                    ),
                    html.Div([html.Strong("Keywords: "), kw], className="cluster-card-keywords"),
                    html.Div([html.Strong("People: "), people], className="cluster-card-keywords"),
                    html.Div(posters, className="cluster-poster-row"),
                ],
            )
        )
    return html.Div(items, className="cluster-cards-grid")


def layout() -> html.Div:
    """Build the taste map page."""
    if not data.has_taste_space():
        return html.Div([
            page_header("Taste map", "Unsupervised view of your watch history in semantic film space."),
            card([
                html.P(
                    "PCA/UMAP and clustering need cached embeddings from ingest or train. "
                    "Run uv run python scripts/main.py --ingest to build the TMDB catalog and map your "
                    "Letterboxd exports (diary / ratings / watched.csv).",
                    style={"color": COLORS["text_secondary"], "margin": 0},
                ),
            ]),
        ])

    year_min, year_max = data.taste_year_bounds()

    filter_toolbar = html.Div(
        className="toolbar",
        children=[
            _toolbar_item(
                "Genre",
                dcc.Dropdown(id="tm-genre", options=data.taste_genre_options(), multi=True, placeholder="All"),
                "toolbar-dd",
            ),
            _toolbar_item(
                "Language",
                dcc.Dropdown(id="tm-lang", options=data.taste_language_options(), multi=True, placeholder="All"),
                "toolbar-dd",
            ),
            _toolbar_item(
                "Year watched",
                dcc.RangeSlider(
                    id="tm-years",
                    min=year_min,
                    max=year_max,
                    step=1,
                    value=[year_min, year_max],
                    tooltip={"placement": "bottom"},
                    allowCross=False,
                ),
                "toolbar-slider",
            ),
            _toolbar_item(
                "Your rating",
                dcc.RangeSlider(
                    id="tm-rating",
                    min=0.5,
                    max=5.0,
                    step=0.5,
                    value=[0.5, 5.0],
                    tooltip={"placement": "bottom"},
                    allowCross=False,
                ),
                "toolbar-slider",
            ),
            html.Div(
                className="toolbar-meta",
                children=[
                    html.Span(id="tm-count", className="filter-count"),
                    html.Span(id="tm-selected", className="filter-selected"),
                ],
            ),
        ],
    )

    map_toolbar = html.Div(
        className="toolbar",
        children=[
            _toolbar_item(
                "Projection",
                dcc.Dropdown(id="tm-projection", options=PROJECTION_OPTIONS, value="pca", clearable=False),
                "toolbar-dd",
            ),
            _toolbar_item(
                "View",
                dcc.RadioItems(
                    id="tm-dimensions",
                    options=DIMENSION_OPTIONS,
                    value=3,
                    inline=True,
                    className="radio-inline",
                ),
                "toolbar-dd",
            ),
            _toolbar_item(
                "Clusters",
                dcc.Slider(id="tm-clusters", min=2, max=12, step=1, value=6, marks={2: "2", 6: "6", 12: "12"}),
                "toolbar-slider",
            ),
            _toolbar_item(
                "Color by",
                dcc.Dropdown(id="tm-color-by", options=COLOR_BY_OPTIONS, value="cluster", clearable=False),
                "toolbar-dd",
            ),
        ],
    )

    return html.Div([
        page_header(
            "Taste map",
            "Main chart: your Letterboxd library. TMDB reference catalog is the 2D chart below for discovery context.",
        ),
        filter_toolbar,
        map_toolbar,
        html.Div(
            className="explorer-grid",
            children=[
                card([dcc.Graph(id="tm-scatter", figure=empty_figure(), config={"displayModeBar": True})]),
                html.Div(
                    className="taste-side-stack",
                    children=[
                        html.Div(
                            id="tm-detail",
                            children=detail_placeholder("Click a point in the map to see film details."),
                        ),
                        card([
                            html.Div("Nearest in embedding space", className="metric-label"),
                            data_table("tm-neighbors", NEIGHBOR_COLUMNS, page_size=8),
                        ]),
                        card([
                            html.Div("Similar — not in your library", className="metric-label"),
                            html.Div(
                                "Discovery picks from the TMDB map, closest in plot space.",
                                className="detail-meta",
                                style={"marginBottom": "8px"},
                            ),
                            data_table("tm-recommendations", RECOMMEND_COLUMNS, page_size=8),
                        ]),
                    ],
                ),
            ],
        ),
        section_title("TMDB reference catalog"),
        card([dcc.Graph(id="tm-reference", figure=empty_figure(), config={"displayModeBar": True})]),
        section_title("Watch timeline"),
        card([dcc.Graph(id="tm-timeline", config={"displayModeBar": False})]),
        section_title("Cluster neighborhoods"),
        card([html.Div(id="tm-cluster-cards")]),
        section_title("Cluster summary"),
        card([data_table("tm-summary", SUMMARY_COLUMNS, sortable=True)]),
    ])


def _filtered_df(
    genres: list[str] | None,
    languages: list[str] | None,
    years: list[int] | None,
    ratings: list[float] | None,
    n_clusters: int,
    projection: str,
) -> pd.DataFrame:
    library = data.library_films()
    filtered = data.filter_films(library, genres=genres, languages=languages, year_range=years, rating_range=ratings)
    return data.films_with_clusters(n_clusters, projection=projection, subset=filtered)


def _click_from_maps(scatter_click, reference_click):
    if callback_context.triggered_id == "tm-reference":
        return reference_click
    return scatter_click


def _click_selection(click_data: dict | None) -> tuple[str | None, int | float | None, str]:
    if not click_data or not click_data.get("points"):
        return None, None, ""
    custom = click_data["points"][0].get("customdata")
    if not custom or len(custom) < 2:
        return None, None, ""
    name, year = custom[0], custom[1]
    return name, year, f"Selected: {name} ({year})"


@callback(
    Output("tm-scatter", "figure"),
    Output("tm-reference", "figure"),
    Output("tm-timeline", "figure"),
    Output("tm-summary", "data"),
    Output("tm-cluster-cards", "children"),
    Output("tm-count", "children"),
    Input("tm-clusters", "value"),
    Input("tm-genre", "value"),
    Input("tm-lang", "value"),
    Input("tm-years", "value"),
    Input("tm-rating", "value"),
    Input("tm-color-by", "value"),
    Input("tm-projection", "value"),
    Input("tm-dimensions", "value"),
)
def update_taste_map(
    n_clusters,
    genres,
    languages,
    years,
    ratings,
    color_by,
    projection,
    dimensions,
):
    if not data.has_taste_space():
        empty = empty_figure("Taste map unavailable")
        return empty, empty, empty, [], html.Div(), ""

    n_clusters = n_clusters or 6
    projection = projection or "pca"
    dimensions = int(dimensions or 3)
    universe = data.get_taste_films()
    library = data.library_films()
    df = _filtered_df(genres, languages, years, ratings, n_clusters, projection)

    rated_n = int(pd.to_numeric(library["Rating"], errors="coerce").notna().sum())
    count = f"Highlighting {len(df)} of {len(library)} library films ({rated_n} rated)"
    if "in_library" in universe.columns and (~universe["in_library"]).any():
        count += f" · {len(universe):,} films on map"

    if df.empty:
        empty = empty_figure()
        return empty, empty, empty, [], _cluster_cards([]), count

    explained = data.get_pca_explained() if projection == "pca" else None
    uirev = f"{projection}-{dimensions}-{color_by or 'cluster'}-{n_clusters}"
    fig = taste_map_figure(
        universe,
        projection=projection,
        dimensions=dimensions,
        explained_variance=explained,
        color_by=color_by or "cluster",
        highlight_df=df,
        uirevision=uirev,
    )
    ref_fig = taste_reference_figure(
        universe,
        projection=projection,
        explained_variance=explained,
        uirevision=f"ref-{uirev}",
    )
    timeline = taste_timeline_figure(df)
    summary_df = cluster_summary(df)
    cards = _cluster_cards(cluster_insights(df))

    return fig, ref_fig, timeline, summary_df.to_dict("records"), cards, count


def _format_pred(value) -> str:
    pred = pd.to_numeric(value, errors="coerce")
    return f"{pred:.2f}" if pd.notna(pred) else "—"


def _short_genres(value: str) -> str:
    parts = [g for g in str(value or "").split("|") if g]
    return ", ".join(parts[:3]) if parts else "—"


@callback(
    Output("tm-selected", "children"),
    Output("tm-neighbors", "data"),
    Output("tm-recommendations", "data"),
    Input("tm-scatter", "clickData"),
    Input("tm-reference", "clickData"),
)
def update_taste_selection(scatter_click, reference_click):
    click_data = _click_from_maps(scatter_click, reference_click)
    """Neighbors and selection label only — do not redraw the map on click/pan."""
    if not data.has_taste_space():
        return "", [], []

    name, year, selected_text = _click_selection(click_data)
    if not name or year is None:
        raise PreventUpdate

    embeddings = data.get_taste_embeddings()
    if embeddings is None:
        return selected_text, [], []

    universe = data.get_taste_films()
    exclude = data.library_film_keys()

    neighbors_rows: list[dict] = []
    nn = nearest_neighbor_films(universe, embeddings, name, year, k=8)
    if not nn.empty:
        nn = nn.copy()
        nn["Rating"] = pd.to_numeric(nn["Rating"], errors="coerce").map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
        neighbors_rows = nn[["Name", "Year", "similarity", "cluster_label", "Rating"]].to_dict("records")

    rec_rows: list[dict] = []
    rec = recommend_similar_films(universe, embeddings, name, year, exclude, k=10)
    if not rec.empty:
        rec = rec.copy()
        if "pred" in rec.columns:
            rec["pred"] = rec["pred"].map(_format_pred)
        else:
            rec["pred"] = "—"
        rec["genres"] = rec["genres"].map(_short_genres)
        rec_rows = rec[["Name", "Year", "similarity", "pred", "genres"]].to_dict("records")

    return selected_text, neighbors_rows, rec_rows


@callback(
    Output("tm-detail", "children"),
    Input("tm-scatter", "clickData"),
    Input("tm-reference", "clickData"),
)
def update_taste_detail(scatter_click, reference_click):
    click_data = _click_from_maps(scatter_click, reference_click)
    if not click_data or not click_data.get("points"):
        raise PreventUpdate

    custom = click_data["points"][0].get("customdata")
    if not custom or len(custom) < 2:
        return detail_placeholder("Click a point in the map to see film details.")

    name, year = custom[0], custom[1]
    tmdb_id = custom[2] if len(custom) > 2 else None
    row = resolve_film_row(data.get_taste_films(), name, year, tmdb_id)
    if row is None:
        return detail_placeholder("Film not found in the current dataset.")
    return film_detail_card(row)


dash.register_page(__name__, path="/taste", name="Taste map", title="Film Recommendations — Taste map", layout=layout)
