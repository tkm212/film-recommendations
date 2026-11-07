"""Shared film detail panel used by Films and Taste map pages."""

from __future__ import annotations

import pandas as pd
from dash import html

from . import data
from .components import card, chip
from .theme import COLORS


def detail_placeholder(message: str = "Select a film to see details.") -> html.Div:
    return card([html.Div(message, className="detail-placeholder")])


def _rating_block(label: str, value: str) -> html.Div:
    return html.Div(
        className="detail-rating",
        children=[html.Div(label, className="metric-label"), html.Div(value, className="detail-rating-value")],
    )


def film_detail_card(row: pd.Series) -> html.Div:
    """Detail card for one film (rated or unwatched-without-rating)."""
    poster = data.poster_url(str(row.get("poster_path", "") or ""), "w342")
    genres = [g for g in str(row.get("genres", "") or "").split("|") if g]

    meta_bits = []
    runtime = row.get("runtime")
    if pd.notna(runtime) and runtime:
        meta_bits.append(f"{int(runtime)} min")
    lang = row.get("original_language")
    if lang:
        meta_bits.append(data.language_name(str(lang)))
    if pd.notna(row.get("Date")):
        meta_bits.append(f"Watched {pd.Timestamp(row['Date']).date()}")

    children: list = []
    if poster:
        children.append(html.Img(src=poster, className="detail-poster", alt=f"{row['Name']} poster"))
    children.extend([
        html.H3(f"{row['Name']} ({row['Year']})", className="detail-title"),
        html.Div(className="detail-chips", children=[chip(g) for g in genres]),
        html.Div(" · ".join(meta_bits), className="detail-meta"),
    ])

    rating = pd.to_numeric(row.get("Rating"), errors="coerce")
    pred = pd.to_numeric(row.get("pred"), errors="coerce")
    err = pd.to_numeric(row.get("err"), errors="coerce")

    rating_rows: list[html.Div] = []
    if pd.notna(rating):
        rating_rows.append(_rating_block("Your rating", f"{rating:.1f}"))
    else:
        rating_rows.append(_rating_block("Your rating", "—"))
    if pd.notna(pred):
        rating_rows.append(_rating_block("Predicted", f"{pred:.2f}"))
    elif pd.notna(rating):
        rating_rows.append(_rating_block("Predicted", "—"))

    if rating_rows:
        children.append(html.Div(className="detail-ratings", children=rating_rows))

    if pd.notna(err):
        err_color = COLORS["danger"] if err > 0 else COLORS["success"]
        verdict = "model over-predicted" if err > 0 else "model under-predicted"
        children.append(html.Div(f"{err:+.2f} — {verdict}", className="detail-err", style={"color": err_color}))

    overview = row.get("overview")
    if isinstance(overview, str) and overview.strip():
        children.append(html.P(overview, className="detail-overview"))

    people = row.get("people_text")
    if isinstance(people, str) and people.strip():
        display = (
            ", ".join(p.strip() for p in people.split("|") if p.strip())
            if "|" in people
            else people
        )
        children.extend([
            html.Div("Director & cast", className="metric-label"),
            html.P(display, className="detail-people"),
        ])

    uri = row.get("Letterboxd URI", "")
    if isinstance(uri, str) and uri.startswith("http"):
        children.append(html.A("View on Letterboxd", href=uri, target="_blank", className="detail-link"))

    cluster = row.get("cluster_label")
    if isinstance(cluster, str) and cluster:
        children.append(html.Div(f"Taste map: {cluster}", className="detail-meta", style={"marginTop": "12px"}))

    return card(children)


def resolve_film_row(df: pd.DataFrame, name: str, year: int | float, tmdb_id: int | float | None = None) -> pd.Series | None:
    """Find a film row by TMDB id or name+year."""
    if tmdb_id is not None and pd.notna(tmdb_id) and "tmdb_id" in df.columns:
        matches = df[df["tmdb_id"] == tmdb_id]
        if not matches.empty:
            return matches.iloc[0]
    year_int = int(year)
    matches = df[(df["Name"] == name) & (pd.to_numeric(df["Year"], errors="coerce") == year_int)]
    if matches.empty:
        return None
    return matches.iloc[0]
