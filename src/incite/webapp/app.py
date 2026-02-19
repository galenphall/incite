"""inCite testing webapp - Streamlit entry point."""

from pathlib import Path

import streamlit as st

from incite.corpus.zotero_reader import find_zotero_data_dir, get_library_stats
from incite.retrieval.factory import get_storage_key
from incite.webapp.state import (
    DEFAULT_EMBEDDER,
    EMBEDDERS,
    extract_and_save_pdfs,
    get_cache_dir,
    get_config,
    get_paper_dict,
    get_paragraph_retriever,
    get_retriever,
    has_chunks,
    load_zotero_chunks,
    load_zotero_direct,
    save_config,
)


def _render_explore_tab(papers, paper_dict, embedder_type: str) -> None:
    """Render the UMAP explore tab in the Streamlit webapp."""
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
        projection = None  # stale cache

    if projection is None:
        if not index_path.exists():
            st.warning("No FAISS index found. Load the Search tab first to build the index.")
            return

        with st.spinner("Computing UMAP projection (this may take 10-30 seconds)..."):
            store = FAISSStore()
            store.load(index_path)
            paper_ids, embeddings = extract_embeddings_from_faiss(store)

            # Build metadata dict
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

    # Controls
    col_color, col_search, _ = st.columns([1, 2, 1])
    with col_color:
        color_field = st.selectbox("Color by", ["year", "journal"], key="umap_color")
    with col_search:
        search_query = st.text_input(
            "Search papers", placeholder="Filter by title or author...", key="umap_search"
        )

    # Search filtering
    highlight_ids = None
    if search_query:
        q = search_query.lower()
        highlight_ids = set()
        for point in projection.points:
            if q in point.title.lower() or q in " ".join(point.authors).lower():
                highlight_ids.add(point.paper_id)
        st.caption(f"{len(highlight_ids)} of {len(projection.points)} papers matched")

    # Build and display figure
    import plotly.graph_objects as go

    fig_dict = build_umap_figure(projection, color_field, highlight_ids)
    fig = go.Figure(fig_dict)
    st.plotly_chart(fig, use_container_width=True, key="umap_chart")


def main():
    st.set_page_config(
        page_title="inCite Testing",
        page_icon="📚",
        layout="wide",
    )

    st.title("inCite Testing UI")

    # Load config
    config = get_config()

    # Sidebar for configuration
    with st.sidebar:
        st.header("Zotero Library")

        # Auto-detect or use saved value
        default_zotero = find_zotero_data_dir()
        saved_dir = config.get("zotero", {}).get("data_dir", "")

        zotero_dir = st.text_input(
            "Zotero data directory",
            value=saved_dir or (str(default_zotero) if default_zotero else ""),
            help="Contains zotero.sqlite and storage/. Usually ~/Zotero on Mac/Linux.",
        )

        # Save if changed
        if zotero_dir != config.get("zotero", {}).get("data_dir", ""):
            config.setdefault("zotero", {})["data_dir"] = zotero_dir
            save_config(config)

        # Validate Zotero directory
        zotero_path = Path(zotero_dir).expanduser() if zotero_dir else None
        zotero_valid = zotero_path and (zotero_path / "zotero.sqlite").exists()

        if zotero_dir and zotero_valid:
            st.success("Zotero database found")
            # Show library stats
            stats = get_library_stats(zotero_path)
            if "error" in stats:
                st.caption(f"Stats unavailable: {stats['error']}")
            else:
                cols = st.columns(3)
                cols[0].metric("Papers", stats["total_papers"])
                cols[1].metric("Abstracts", stats["with_abstracts"])
                cols[2].metric("PDFs", stats["with_pdfs"])
        elif zotero_dir:
            st.error("zotero.sqlite not found")

        st.divider()

        # Method selection
        method = st.selectbox(
            "Retrieval method",
            options=["hybrid", "neural", "bm25"],
            index=["hybrid", "neural", "bm25"].index(
                config.get("webapp", {}).get("default_method", "hybrid")
            ),
            help="hybrid combines neural embeddings with BM25 keyword matching",
        )

        # Embedder selection (only relevant for neural/hybrid)
        embedder_options = list(EMBEDDERS.keys())
        embedder_labels = {k: v["name"] for k, v in EMBEDDERS.items()}

        embedder_type = st.selectbox(
            "Embedding model",
            options=embedder_options,
            index=embedder_options.index(
                config.get("webapp", {}).get("default_embedder", DEFAULT_EMBEDDER)
            ),
            format_func=lambda x: embedder_labels.get(x, x),
            help="MiniLM is faster; SPECTER2 is specialized for scientific papers",
            disabled=(method == "bm25"),
        )

        # Top-k selection
        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=20,
            value=config.get("webapp", {}).get("default_k", 5),
        )

        st.divider()

        # Refresh button
        refresh = st.button("Refresh Corpus & Index", type="secondary")

        st.divider()

        # Status info
        st.caption("Cache location: ~/.incite/")

    # Check if Zotero path is valid
    if not zotero_dir:
        st.warning("Please set your Zotero data directory in the sidebar.")
        st.info(
            "**Where to find it:**\n"
            "- **Mac/Linux:** Usually `~/Zotero`\n"
            "- **Windows:** Usually `C:\\Users\\YourName\\Zotero`\n\n"
            "The directory should contain `zotero.sqlite` and a `storage/` folder."
        )
        return

    if not zotero_valid:
        st.error(
            f"Could not find `zotero.sqlite` in: {zotero_dir}\n\n"
            "Make sure this is your Zotero data directory, not the application folder."
        )
        return

    # Load corpus with progress
    status_container = st.empty()

    def update_status(msg):
        status_container.info(msg)

    # Use session state to cache corpus
    if "papers" not in st.session_state or refresh:
        with st.spinner("Loading corpus from Zotero (may take a moment)..."):
            try:
                papers = load_zotero_direct(
                    zotero_path,
                    force_refresh=refresh,
                    progress_callback=update_status,
                )
                st.session_state.papers = papers
                st.session_state.paper_dict = get_paper_dict(papers)
            except Exception as e:
                import sqlite3

                if isinstance(e, sqlite3.OperationalError) and "locked" in str(e).lower():
                    st.error("Zotero database is locked. Close Zotero and reload this page.")
                elif isinstance(e, PermissionError):
                    st.error("Permission denied reading Zotero library. Check file permissions.")
                else:
                    st.error(f"Error loading corpus: {e}")
                return

    papers = st.session_state.papers
    paper_dict = st.session_state.paper_dict
    status_container.empty()

    # Show corpus stats in sidebar
    with st.sidebar:
        st.metric("Papers in corpus", len(papers))
        with_abstract = sum(1 for p in papers if p.abstract)
        st.metric("With abstracts", with_abstract)

        # Check if paragraph mode is available
        can_use_paragraphs = has_chunks(papers)
        if can_use_paragraphs:
            with_full_text = sum(1 for p in papers if p.full_text or p.paragraphs)
            st.metric("With full text", with_full_text)

    # Show paragraph mode toggle and PDF extraction
    use_paragraph_mode = False
    with st.sidebar:
        st.divider()
        st.subheader("Paragraph Search")

        if can_use_paragraphs:
            use_paragraph_mode = st.checkbox(
                "Enable paragraph-level search",
                value=st.session_state.get("paragraph_mode_enabled", False),
                key="paragraph_mode_enabled",
                help="Search within paper content to show matched passages",
            )
            # Paragraph display options (only shown when paragraph mode enabled)
            if use_paragraph_mode:
                force_show_paragraphs = st.checkbox(
                    "Always show matched paragraphs",
                    value=st.session_state.get("force_show_paragraphs", False),
                    key="force_show_paragraphs",
                    help="Override adaptive display and always show matched text",
                )
                st.slider(
                    "Paragraph display threshold",
                    min_value=0.3,
                    max_value=0.9,
                    value=0.65,
                    step=0.05,
                    key="para_threshold",
                    help="Show paragraph when chunk score exceeds this threshold",
                    disabled=force_show_paragraphs,
                )
        else:
            st.info("Extract PDFs to enable paragraph search")

        # Show how many papers have PDFs
        papers_with_pdfs = sum(1 for p in papers if p.source_file)
        st.caption(f"{papers_with_pdfs} papers have linked PDFs")

        # Extract PDFs button (uses source_file paths directly)
        extract_pdfs = st.button(
            "Extract & Embed PDFs",
            type="primary" if not can_use_paragraphs else "secondary",
            help="Extract text from PDFs and build paragraph embeddings",
            disabled=papers_with_pdfs == 0,
        )

        if extract_pdfs:
            # Create progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()

            def pdf_progress(current, total, message):
                if total > 0:
                    progress_bar.progress(current / total)
                status_text.text(message)

            try:
                stats = extract_and_save_pdfs(
                    papers,
                    progress_callback=pdf_progress,
                )
                progress_bar.empty()
                status_text.empty()
                st.success(
                    f"Extracted text from {stats['extracted']} PDFs "
                    f"(found {stats['found_pdfs']} of {stats['with_path']} with paths)"
                )
                # Clear session state to force reload
                for key in list(st.session_state.keys()):
                    if key not in ["paragraph_mode_enabled"]:
                        del st.session_state[key]
                st.rerun()
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error extracting PDFs: {e}")

    # Build/load retriever
    if use_paragraph_mode and can_use_paragraphs:
        # Paragraph-level retrieval
        chunks_key = "chunks"
        if chunks_key not in st.session_state or refresh:
            n_ft = sum(1 for p in papers if p.full_text or p.paragraphs)
            with st.spinner(f"Loading chunks from {n_ft} full-text papers..."):
                try:
                    chunks = load_zotero_chunks(
                        papers,
                        force_rebuild=refresh,
                        progress_callback=update_status,
                    )
                    st.session_state[chunks_key] = chunks
                except Exception as e:
                    st.error(f"Error loading chunks: {e}")
                    return

        chunks = st.session_state[chunks_key]

        # Key includes embedder and method
        retriever_key = f"paragraph_retriever_{method}_{embedder_type}"
        if retriever_key not in st.session_state or refresh:
            embedder_name = EMBEDDERS.get(embedder_type, {}).get("name", embedder_type)
            # Use progress bar for chunk embedding (can be very slow)
            para_progress_bar = st.progress(0)
            para_status_text = st.empty()

            def para_progress(current_or_msg, total=None, message=None):
                if total is not None:
                    # Called as (current, total, message) from embedding
                    para_progress_bar.progress(current_or_msg / total)
                    para_status_text.text(message)
                else:
                    # Called as (message,) from other steps
                    para_status_text.text(current_or_msg)

            try:
                retriever = get_paragraph_retriever(
                    chunks,
                    papers,
                    method=method,
                    embedder_type=embedder_type,
                    force_rebuild=refresh,
                    progress_callback=para_progress,
                )
                st.session_state[retriever_key] = retriever
                para_progress_bar.empty()
                para_status_text.empty()
            except Exception as e:
                para_progress_bar.empty()
                para_status_text.empty()
                st.error(f"Error loading paragraph retriever: {e}")
                return

        retriever = st.session_state[retriever_key]
    else:
        # Paper-level retrieval
        retriever_key = f"retriever_{method}_{embedder_type}"
        if retriever_key not in st.session_state or refresh:
            embedder_name = EMBEDDERS.get(embedder_type, {}).get("name", embedder_type)
            with st.spinner(
                f"Loading {method} retriever with {embedder_name} ({len(papers)} papers)..."
            ):
                try:
                    retriever = get_retriever(
                        papers,
                        method=method,
                        embedder_type=embedder_type,
                        force_rebuild=refresh,
                        progress_callback=update_status,
                    )
                    st.session_state[retriever_key] = retriever
                except Exception as e:
                    st.error(f"Error loading retriever: {e}")
                    return

        retriever = st.session_state[retriever_key]

    status_container.empty()

    # Tabs: Search and Explore
    tab1, tab2 = st.tabs(["Search", "Explore"])

    with tab2:
        _render_explore_tab(papers, paper_dict, embedder_type)

    # Main content area (Search tab)
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Your Text")

            # Text input
            query_text = st.text_area(
                "Enter text to get citation recommendations for:",
                height=200,
                placeholder="Paste your draft text here...",
                help="Suggest papers from your Zotero library as citations for this text.",
            )

            # Context extraction help
            with st.expander("Tips for best results"):
                st.markdown(
                    """
                - **Local context works best**: 1-3 sentences around the citation
                - **Be specific**: Include technical terms and key concepts
                - **Include surrounding context**: Helps the model understand your topic
                """
                )

            # Get recommendations button
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
                    # Get display settings (only relevant for paragraph mode)
                    force_paragraphs = st.session_state.get("force_show_paragraphs", False)
                    threshold = (
                        st.session_state.get("para_threshold", 0.65) if use_paragraph_mode else 0.65
                    )

                    for i, result in enumerate(results, 1):
                        paper = paper_dict.get(result.paper_id)
                        if paper:
                            # Confidence badge coloring
                            conf = result.confidence
                            if conf >= 0.55:
                                conf_color = "green"
                            elif conf >= 0.35:
                                conf_color = "orange"
                            else:
                                conf_color = "red"

                            # Determine display mode using adaptive logic
                            display_mode = "paper"
                            if use_paragraph_mode and result.matched_paragraph:
                                if force_paragraphs:
                                    display_mode = "paragraph"
                                else:
                                    display_mode = result.get_display_mode(para_threshold=threshold)

                            # Paper display
                            with st.container():
                                # Title with confidence and mode indicator
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

                                # Show matched paragraph based on display mode
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

                                # Abstract preview
                                if paper.abstract:
                                    preview = paper.abstract[:300]
                                    if len(paper.abstract) > 300:
                                        preview += "..."
                                    with st.expander("Abstract"):
                                        st.write(paper.abstract)

                                # BibTeX key for easy copying
                                if paper.bibtex_key:
                                    st.code(paper.bibtex_key, language=None)

                                st.divider()

            elif not get_recs:
                st.info(
                    "Enter some text and click 'Get Recommendations' to see suggested citations."
                )


if __name__ == "__main__":
    main()
