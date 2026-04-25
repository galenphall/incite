"""Tab implementations for the inCite Streamlit webapp.

Contains the rendering logic for the Search and Explore tabs.
Called by the main app layout in webapp/app.py.

Related modules:
    - incite.webapp.app: Main Streamlit app entry point and layout.
    - incite.webapp.state: State management and retriever setup.
"""

from typing import Any

import streamlit as st

from incite.retrieval.factory import get_storage_key
from incite.webapp.state import get_cache_dir


def _render_explore_tab(papers: list, paper_dict: dict, embedder_type: str) -> None:
    """Render the UMAP explore tab in the Streamlit webapp.

    Loads or computes a UMAP projection of paper embeddings from the FAISS
    index and displays an interactive Plotly scatter plot. Supports color
    encoding by year or journal and text-based paper filtering.

    Args:
        papers: List of Paper objects in the current corpus.
        paper_dict: Mapping from paper_id to Paper objects.
        embedder_type: Key identifying the embedder (e.g. "minilm-ft").
    """
    from incite.embeddings.stores import FAISSStore
    from incite.visualization.plotly_charts import build_umap_figure
    from incite.visualization.umap_projection import (
        compute_umap_projection,
        extract_embeddings_from_faiss,
        load_projection_cache,
        save_projection_cache,
    )

    cache_dir = get_cache_dir()
    storage_key = get_storage_key(embedder_type)
    cache_path = cache_dir / f"umap_{storage_key}.json"
    index_path = cache_dir / f"zotero_index_{storage_key}"

    # Load or compute projection
    projection = load_projection_cache(cache_path)
    if projection is not None and projection.num_papers != len(papers):
        projection = None  # stale cache — paper count changed

    if projection is None:
        if not index_path.exists():
            st.warning("No FAISS index found. Load the Search tab first to build the index.")
            return

        with st.spinner("Computing UMAP projection (this may take 10-30 seconds)..."):
            store = FAISSStore()
            store.load(index_path)
            paper_ids, embeddings = extract_embeddings_from_faiss(store)

            # Build metadata dict for each paper in the index
            paper_metadata = {}
            for pid in paper_ids:
                p = paper_dict.get(pid)
                if p:
                    paper_metadata[pid] = {
                        "title": p.title,
                        "authors": list(p.authors) if p.authors else [],
                        "year": p.year,
                        "journal": p.journal,
                    }

            projection = compute_umap_projection(
                paper_ids, embeddings, paper_metadata, embedder_type
            )
            save_projection_cache(projection, cache_path)

    # Controls: color field and search filter
    col_color, col_search, _ = st.columns([1, 2, 1])
    with col_color:
        color_field = st.selectbox("Color by", ["year", "journal"], key="umap_color")
    with col_search:
        search_query = st.text_input(
            "Search papers", placeholder="Filter by title or author...", key="umap_search"
        )

    # Search filtering: highlight matching paper IDs
    highlight_ids = None
    if search_query:
        q = search_query.lower()
        highlight_ids = set()
        for point in projection.points:
            if q in point.title.lower() or q in " ".join(point.authors).lower():
                highlight_ids.add(point.paper_id)
        st.caption(f"{len(highlight_ids)} of {len(projection.points)} papers matched")

    # Build and display the interactive Plotly figure
    import plotly.graph_objects as go

    fig_dict = build_umap_figure(projection, color_field, highlight_ids)
    fig = go.Figure(fig_dict)
    st.plotly_chart(fig, use_container_width=True, key="umap_chart")


def _render_search_tab(
    retriever: Any,
    paper_dict: dict,
    top_k: int,
    use_paragraph_mode: bool,
) -> None:
    """Render the Search tab in the Streamlit webapp.

    Displays a text input area on the left and citation recommendations on
    the right. Each result shows the paper title, authors, year, confidence
    badge, and optionally matched paragraphs (when paragraph mode is active).

    Confidence badge coloring:
        - green  : confidence >= 0.55 (high confidence)
        - orange : confidence >= 0.35 (moderate confidence)
        - red    : confidence < 0.35  (low confidence)

    Display mode selection (paragraph mode only):
        - "paragraph"           : single best passage shown inline
        - "paper_with_summary"  : multiple passages shown in expander
        - "paper"               : abstract-only (chunk score below threshold)

    Args:
        retriever: An instantiated Retriever (paper-level or paragraph-level).
        paper_dict: Mapping from paper_id to Paper objects.
        top_k: Maximum number of results to return.
        use_paragraph_mode: Whether paragraph-level retrieval is active.
    """
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Your Text")

        query_text = st.text_area(
            "Enter text to get citation recommendations for:",
            height=200,
            placeholder="Paste your draft text here...",
            help="Suggest papers from your Zotero library as citations for this text.",
        )

        # Usage tips
        with st.expander("Tips for best results"):
            st.markdown(
                """
            - **Local context works best**: 1-3 sentences around the citation
            - **Be specific**: Include technical terms and key concepts
            - **Include surrounding context**: Helps the model understand your topic
            """
            )

        get_recs = st.button(
            "Get Recommendations",
            type="primary",
            disabled=not query_text.strip(),
        )

    with col2:
        st.subheader("Recommendations")

        if get_recs and query_text.strip():
            with st.spinner("Searching..."):
                results = retriever.retrieve(
                    query_text, k=top_k, papers=paper_dict, deduplicate=True
                )

            if not results:
                st.warning("No results found.")
            else:
                # Retrieve display settings set in the sidebar (paragraph mode only)
                force_paragraphs = st.session_state.get("force_show_paragraphs", False)
                threshold = (
                    st.session_state.get("para_threshold", 0.65) if use_paragraph_mode else 0.65
                )

                for i, result in enumerate(results, 1):
                    paper = paper_dict.get(result.paper_id)
                    if paper:
                        # Confidence badge: color encodes retrieval certainty
                        conf = result.confidence
                        if conf >= 0.55:
                            conf_color = "green"
                        elif conf >= 0.35:
                            conf_color = "orange"
                        else:
                            conf_color = "red"

                        # Determine display mode using adaptive logic.
                        # "paragraph" shows the best matched chunk inline;
                        # "paper_with_summary" shows it in a collapsible expander;
                        # "paper" falls back to abstract-only.
                        display_mode = "paper"
                        if use_paragraph_mode and result.matched_paragraph:
                            if force_paragraphs:
                                display_mode = "paragraph"
                            else:
                                display_mode = result.get_display_mode(para_threshold=threshold)

                        with st.container():
                            # Title with inline confidence badge and mode indicator
                            mode_badge = ""
                            if display_mode == "paragraph":
                                mode_badge = " 📄"
                            elif display_mode == "paper_with_summary":
                                num_chunks = result.score_breakdown.get("num_chunks_matched", 1)
                                mode_badge = f" ({num_chunks} passages)"

                            st.markdown(
                                f"**{i}. {paper.title}**{mode_badge} "
                                f"<span style='color:{conf_color};'"
                                f" title='Confidence: {conf:.2f}'>"
                                f"({conf:.2f})</span>",
                                unsafe_allow_html=True,
                            )

                            # Authors and year
                            if paper.authors:
                                authors_str = ", ".join(paper.authors[:3])
                                if len(paper.authors) > 3:
                                    authors_str += " et al."
                                year_str = f" ({paper.year})" if paper.year else ""
                                st.caption(f"{authors_str}{year_str}")

                            # Matched paragraph display varies by display mode
                            if display_mode == "paragraph" and result.matched_paragraph:
                                chunk_score = result.score_breakdown.get("best_chunk_score", 0)
                                st.info(f"**Matched passage** (score: {chunk_score:.2f})")
                                st.markdown(f"_{result.matched_paragraph}_")
                            elif (
                                display_mode == "paper_with_summary"
                                and result.matched_paragraph
                            ):
                                num_chunks = result.score_breakdown.get("num_chunks_matched", 1)
                                with st.expander(f"Best of {num_chunks} matched passages"):
                                    st.markdown(f"_{result.matched_paragraph}_")
                            elif result.matched_paragraph and use_paragraph_mode:
                                with st.expander("Matched passage (low confidence)"):
                                    st.markdown(f"_{result.matched_paragraph}_")

                            # Abstract preview (collapsed by default)
                            if paper.abstract:
                                with st.expander("Abstract"):
                                    st.write(paper.abstract)

                            # BibTeX key for easy copying into a manuscript
                            if paper.bibtex_key:
                                st.code(paper.bibtex_key, language=None)

                            st.divider()

        elif not get_recs:
            st.info(
                "Enter some text and click 'Get Recommendations' to see suggested citations."
            )
