"""Plotly chart builders for UMAP visualization."""

from __future__ import annotations

from typing import Optional

from incite.visualization.umap_projection import UMAPProjection


def build_umap_figure(
    projection: UMAPProjection,
    color_field: str = "year",
    highlight_ids: Optional[set[str]] = None,
) -> dict:
    """Build a Plotly figure dict for a UMAP projection.

    Args:
        projection: The UMAP projection to visualize.
        color_field: Field to color by ("year", "journal", or "collection").
        highlight_ids: If set, highlight these paper IDs and dim others.

    Returns:
        A Plotly figure dict (JSON-serializable).
    """
    import plotly.graph_objects as go

    points = projection.points

    x = [p.x for p in points]
    y = [p.y for p in points]
    ids = [p.paper_id for p in points]

    # Build hover text
    hover_text = []
    for p in points:
        authors_str = ", ".join(p.authors[:3])
        if len(p.authors) > 3:
            authors_str += " et al."
        year_str = str(p.year) if p.year else ""
        journal_str = p.journal or ""
        hover_text.append(
            f"<b>{p.title}</b><br>"
            f"{authors_str}<br>"
            f"{year_str}" + (f" · {journal_str}" if journal_str else "")
        )

    # Determine sizes and opacities for highlighting
    if highlight_ids is not None:
        sizes = [12 if pid in highlight_ids else 4 for pid in ids]
        opacities = [1.0 if pid in highlight_ids else 0.2 for pid in ids]
    else:
        sizes = [7] * len(points)
        opacities = [0.8] * len(points)

    # Build color values
    if color_field == "year":
        color_values = [p.year for p in points]
        marker = dict(
            size=sizes,
            opacity=opacities,
            color=color_values,
            colorscale="Viridis",
            colorbar=dict(title="Year", thickness=15, len=0.5),
            line=dict(width=0),
        )
    else:
        # Categorical coloring (journal or collection)
        if color_field == "journal":
            categories = [p.journal or "Unknown" for p in points]
        else:
            categories = [p.journal or "Unknown" for p in points]

        # Assign color indices
        unique_cats = sorted(set(categories))
        cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
        color_values = [cat_to_idx[c] for c in categories]

        # Use a qualitative palette
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#aec7e8",
            "#ffbb78",
            "#98df8a",
            "#ff9896",
            "#c5b0d5",
        ]
        colors = [palette[idx % len(palette)] for idx in color_values]
        marker = dict(
            size=sizes,
            opacity=opacities,
            color=colors,
            line=dict(width=0),
        )

    fig = go.Figure(
        data=[
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=marker,
                text=hover_text,
                hoverinfo="text",
                customdata=ids,
            )
        ],
        layout=go.Layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=20, b=20),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="IBM Plex Sans, sans-serif",
            ),
        ),
    )

    return fig.to_dict()
