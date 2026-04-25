"""inCite Streamlit webapp — main entry point and layout.

Handles sidebar configuration, corpus loading, retriever setup, and tab
layout. Tab rendering is delegated to webapp/app_tabs.py.

Related modules:
    - incite.webapp.app_tabs: Search and Explore tab implementations.
    - incite.webapp.state: Session state helpers, corpus loading, retriever factories.
"""

from pathlib import Path

import streamlit as st

from incite.corpus.zotero_reader import find_zotero_data_dir, get_library_stats
from incite.webapp.app_tabs import _render_explore_tab, _render_search_tab
from incite.webapp.state import (
    DEFAULT_EMBEDDER,
    EMBEDDERS,
    extract_and_save_pdfs,
    get_config,
    get_paper_dict,
    get_paragraph_retriever,
    get_retriever,
    has_chunks,
    load_zotero_chunks,
    load_zotero_direct,
    save_config,
)


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
        from incite.retrieval.factory import get_available_embedders

        _available = get_available_embedders()
        embedder_options = list(_available.keys())
        embedder_labels = {k: v["name"] for k, v in _available.items()}

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

    with tab1:
        _render_search_tab(retriever, paper_dict, top_k, use_paragraph_mode)


if __name__ == "__main__":
    main()
