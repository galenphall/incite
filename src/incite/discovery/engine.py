"""Discovery engine — expand a library through citation graph, bib coupling,
S2 recommendations, and author works.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Callable

import requests

from incite.corpus.openalex import OpenAlexClient
from incite.corpus.semantic_scholar import SemanticScholarClient
from incite.discovery.models import DiscoveryCandidate

if TYPE_CHECKING:
    from incite.discovery.citation_graph import CitationGraphStore

logger = logging.getLogger(__name__)

# Type for the progress callback: (stage, current, total, message)
ProgressCallback = Callable[[str, int, int, str], None]

# Default configuration
DEFAULT_CONFIG = {
    "n_seed": 30,
    "top_k": 200,
    "n_recs": 10,
    "max_core_refs": 40,
    "skip_recs": False,
    "skip_authors": False,
    "skip_bibcoupling": False,
    "skip_pagerank": False,
    "skip_cocitation": False,
}


class DiscoveryEngine:
    """Full discovery pipeline: resolve, citations, bib coupling, recs, authors, score."""

    def __init__(
        self,
        s2_client: SemanticScholarClient,
        oa_client: OpenAlexClient,
        citation_graph: CitationGraphStore | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        self.s2 = s2_client
        self.oa = oa_client
        self.graph = citation_graph
        self._progress = progress_callback or (lambda *_: None)

    def run(
        self,
        library_papers: list[dict],
        all_papers: list[dict] | None = None,
        config: dict | None = None,
    ) -> list[DiscoveryCandidate]:
        """Execute the full discovery pipeline.

        Args:
            library_papers: Papers from the user's library (dicts with doi, title, authors, year).
            all_papers: Full corpus for author frequency counting. Defaults to library_papers.
            config: Override default settings (n_seed, n_recs, max_core_refs, skip_*).

        Returns:
            Ranked list of DiscoveryCandidate objects (highest score first).
        """
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        all_papers = all_papers or library_papers

        # Filter to papers with DOIs, sort by recency
        papers_with_doi = [p for p in library_papers if p.get("doi")]
        papers_with_doi.sort(key=lambda p: p.get("year") or 0, reverse=True)
        seed_papers = papers_with_doi[: cfg["n_seed"]]

        library_titles = {p.get("title", "").lower().strip() for p in all_papers if p.get("title")}

        # Step 1: Resolve S2 IDs
        self._progress("resolving", 0, len(seed_papers), "Resolving Semantic Scholar IDs...")
        doi_to_s2 = self._resolve_s2_ids(seed_papers)
        library_s2_ids = set(doi_to_s2.values())

        # Step 2: Citation expansion
        self._progress("citations", 0, len(doi_to_s2), "Expanding citation graph...")
        candidates, ref_graph = self._expand_citations(doi_to_s2, seed_papers, library_s2_ids)

        # Step 3: Bibliographic coupling
        if not cfg["skip_bibcoupling"]:
            self._progress("bibcoupling", 0, 0, "Computing bibliographic coupling...")
            self._expand_bibliographic_coupling(
                ref_graph, library_s2_ids, candidates, max_core_refs=cfg["max_core_refs"]
            )

        # Step 3b: PageRank
        if not cfg["skip_pagerank"] and self.graph is not None:
            self._progress("pagerank", 0, 0, "Computing PageRank scores...")
            self._compute_pagerank(library_s2_ids, candidates)

        # Step 3c: Co-citation analysis
        if not cfg["skip_cocitation"] and self.graph is not None:
            self._progress("cocitation", 0, 0, "Computing co-citation scores...")
            self._compute_cocitation(library_s2_ids, candidates)

        # Step 4: S2 Recommendations
        if not cfg["skip_recs"]:
            self._progress("recommendations", 0, cfg["n_recs"], "Fetching S2 recommendations...")
            self._expand_recommendations(
                doi_to_s2, seed_papers, library_s2_ids, candidates, n_recs=cfg["n_recs"]
            )

        # Step 5: Author expansion
        if not cfg["skip_authors"]:
            self._progress("authors", 0, 0, "Expanding author works...")
            self._expand_authors(all_papers, library_s2_ids, candidates, library_titles)

        # Rank and return
        ranked = sorted(candidates.values(), key=lambda c: c.discovery_score, reverse=True)
        self._progress("done", len(ranked), len(ranked), f"Found {len(ranked)} candidates")
        return ranked[: cfg["top_k"]]

    # ------------------------------------------------------------------
    # Step 1: Resolve S2 IDs
    # ------------------------------------------------------------------

    def _resolve_s2_ids(self, papers: list[dict]) -> dict[str, str]:
        """Map library papers to S2 paper IDs via DOI batch lookup.

        Returns:
            dict mapping lowercase DOI -> S2 paper ID
        """
        dois = [p["doi"] for p in papers if p.get("doi")]
        doi_ids = [f"DOI:{doi}" for doi in dois]

        results = self.s2.get_papers_batch(doi_ids)

        doi_to_s2: dict[str, str] = {}
        for doi_id, paper in results.items():
            raw_doi = doi_id.replace("DOI:", "")
            doi_to_s2[raw_doi.lower()] = paper.id

        logger.info("Resolved %d/%d papers to S2 IDs", len(doi_to_s2), len(doi_ids))
        self._progress(
            "resolving", len(doi_to_s2), len(doi_ids), f"Resolved {len(doi_to_s2)} papers"
        )
        return doi_to_s2

    # ------------------------------------------------------------------
    # Step 2: Citation expansion
    # ------------------------------------------------------------------

    def _expand_citations(
        self,
        doi_to_s2: dict[str, str],
        papers: list[dict],
        library_s2_ids: set[str],
    ) -> tuple[dict[str, DiscoveryCandidate], dict[str, set[str]]]:
        """Fetch references and citations for each seed paper.

        Uses the local citation graph when available, falling back to S2 API.

        Returns:
            candidates: s2_id -> DiscoveryCandidate
            ref_graph: ref_s2_id -> set of library s2_ids that cite it
        """
        if self.graph is not None:
            return self._expand_citations_local(doi_to_s2, papers, library_s2_ids)
        return self._expand_citations_api(doi_to_s2, papers, library_s2_ids)

    def _expand_citations_api(
        self,
        doi_to_s2: dict[str, str],
        papers: list[dict],
        library_s2_ids: set[str],
    ) -> tuple[dict[str, DiscoveryCandidate], dict[str, set[str]]]:
        """Citation expansion via S2 API (original approach)."""
        candidates: dict[str, DiscoveryCandidate] = {}
        ref_graph: dict[str, set[str]] = defaultdict(set)

        papers_with_s2 = [
            (p, doi_to_s2[p["doi"].lower()])
            for p in papers
            if p.get("doi") and p["doi"].lower() in doi_to_s2
        ]

        total = len(papers_with_s2)

        for i, (lib_paper, s2_id) in enumerate(papers_with_s2):
            lib_title = lib_paper.get("title", "?")[:60]
            self._progress("citations", i + 1, total, f"Citations: {lib_title}...")

            refs = self.s2.get_paper_references(s2_id, limit=200)
            cites = self.s2.get_paper_citations(s2_id, limit=200)

            # Build reference graph
            for ref in refs:
                if ref.id:
                    ref_graph[ref.id].add(s2_id)

            for p in refs + cites:
                if p.id in library_s2_ids:
                    continue

                if p.id not in candidates:
                    candidates[p.id] = DiscoveryCandidate(
                        s2_id=p.id,
                        title=p.title,
                        authors=[a for a in p.authors if a],
                        year=p.year,
                        doi=p.doi,
                        abstract=p.abstract or "",
                        pdf_url=p.pdf_url,
                        venue=p.journal,
                    )

                cand = candidates[p.id]
                if s2_id not in cand.citing_library_ids:
                    cand.citing_library_ids.append(s2_id)
                    cand.citation_overlap += 1

        logger.info(
            "Citation expansion: %d candidates, %d unique refs tracked",
            len(candidates),
            len(ref_graph),
        )
        return candidates, ref_graph

    def _expand_citations_local(
        self,
        doi_to_s2: dict[str, str],
        papers: list[dict],
        library_s2_ids: set[str],
    ) -> tuple[dict[str, DiscoveryCandidate], dict[str, set[str]]]:
        """Citation expansion via local graph + batch metadata fetch."""
        assert self.graph is not None
        ref_graph: dict[str, set[str]] = defaultdict(set)
        # Track overlap counts: candidate_s2_id -> list of library s2_ids that link to it
        overlap_tracker: dict[str, list[str]] = defaultdict(list)

        papers_with_s2 = [
            (p, doi_to_s2[p["doi"].lower()])
            for p in papers
            if p.get("doi") and p["doi"].lower() in doi_to_s2
        ]

        total = len(papers_with_s2)

        for i, (lib_paper, s2_id) in enumerate(papers_with_s2):
            lib_title = lib_paper.get("title", "?")[:60]
            self._progress("citations", i + 1, total, f"Citations (local): {lib_title}...")

            ref_ids = self.graph.get_references(s2_id)
            cite_ids = self.graph.get_citations(s2_id)

            # Build reference graph
            for ref_id in ref_ids:
                ref_graph[ref_id].add(s2_id)

            # Track overlap for all neighbors
            for neighbor_id in ref_ids + cite_ids:
                if neighbor_id not in library_s2_ids:
                    if s2_id not in overlap_tracker[neighbor_id]:
                        overlap_tracker[neighbor_id].append(s2_id)

        # Rank by overlap count, take top candidates for metadata fetch
        sorted_candidates = sorted(overlap_tracker.items(), key=lambda x: len(x[1]), reverse=True)
        top_ids = [s2_id for s2_id, _ in sorted_candidates[:500]]

        # Batch-fetch metadata from S2 API
        self._progress("citations", total, total, "Fetching metadata for top candidates...")
        batch_results = self.s2.get_papers_batch(top_ids) if top_ids else {}

        # Build DiscoveryCandidate objects from batch results
        candidates: dict[str, DiscoveryCandidate] = {}
        for s2_id, paper in batch_results.items():
            citing_ids = overlap_tracker.get(s2_id, [])
            candidates[s2_id] = DiscoveryCandidate(
                s2_id=s2_id,
                title=paper.title,
                authors=[a for a in paper.authors if a],
                year=paper.year,
                doi=paper.doi,
                abstract=paper.abstract or "",
                citation_overlap=len(citing_ids),
                citing_library_ids=citing_ids,
                pdf_url=paper.pdf_url,
                venue=paper.journal,
            )

        logger.info(
            "Citation expansion (local): %d candidates (from %d graph hits), %d unique refs",
            len(candidates),
            len(overlap_tracker),
            len(ref_graph),
        )
        return candidates, ref_graph

    # ------------------------------------------------------------------
    # Step 3: Bibliographic coupling
    # ------------------------------------------------------------------

    def _expand_bibliographic_coupling(
        self,
        ref_graph: dict[str, set[str]],
        library_s2_ids: set[str],
        candidates: dict[str, DiscoveryCandidate],
        max_core_refs: int = 40,
    ) -> None:
        """Find papers that cite the same references as library papers (in-place).

        Uses the local citation graph when available, falling back to S2 API.
        """
        if not ref_graph:
            return

        if self.graph is not None:
            self._expand_bibcoupling_local(ref_graph, library_s2_ids, candidates)
        else:
            self._expand_bibcoupling_api(ref_graph, library_s2_ids, candidates, max_core_refs)

    def _expand_bibcoupling_api(
        self,
        ref_graph: dict[str, set[str]],
        library_s2_ids: set[str],
        candidates: dict[str, DiscoveryCandidate],
        max_core_refs: int = 40,
    ) -> None:
        """Bibliographic coupling via S2 API (original approach)."""
        # Prioritize references cited by multiple library papers
        multi_cited = {rid: lids for rid, lids in ref_graph.items() if len(lids) >= 2}
        single_cited = {rid: lids for rid, lids in ref_graph.items() if len(lids) == 1}

        sorted_multi = sorted(multi_cited.items(), key=lambda x: len(x[1]), reverse=True)

        candidate_ids = set(candidates.keys())
        single_in_candidates = [
            (rid, lids) for rid, lids in single_cited.items() if rid in candidate_ids
        ]
        single_not_in = [
            (rid, lids) for rid, lids in single_cited.items() if rid not in candidate_ids
        ]

        core_refs = sorted_multi[:]
        remaining = max_core_refs - len(core_refs)
        if remaining > 0:
            core_refs.extend(single_in_candidates[:remaining])
        remaining = max_core_refs - len(core_refs)
        if remaining > 0:
            core_refs.extend(single_not_in[:remaining])
        core_refs = core_refs[:max_core_refs]

        ref_weights = {rid: len(lids) for rid, lids in core_refs}
        total_weight = sum(ref_weights.values())

        citer_weighted_scores: dict[str, float] = defaultdict(float)
        citer_ref_sets: dict[str, set[str]] = defaultdict(set)
        citer_metadata: dict[str, dict] = {}

        for i, (ref_id, _lib_ids) in enumerate(core_refs):
            weight = ref_weights[ref_id]
            self._progress(
                "bibcoupling",
                i + 1,
                len(core_refs),
                f"Bib coupling: core ref {i + 1}/{len(core_refs)}",
            )

            citers = self.s2.get_paper_citations(ref_id, limit=200)

            for citer in citers:
                if not citer.id or citer.id in library_s2_ids:
                    continue

                citer_ref_sets[citer.id].add(ref_id)
                citer_weighted_scores[citer.id] += weight

                if citer.id not in citer_metadata:
                    citer_metadata[citer.id] = {
                        "title": citer.title,
                        "authors": [a for a in citer.authors if a],
                        "year": citer.year,
                        "doi": citer.doi,
                        "abstract": citer.abstract or "",
                        "pdf_url": citer.pdf_url,
                        "venue": citer.journal,
                    }

        # Score candidates by weighted bibliographic coupling
        saturation = max(total_weight * 0.2, 1)

        for citer_id, weighted_score in citer_weighted_scores.items():
            n_refs = len(citer_ref_sets[citer_id])
            if n_refs < 2:
                continue

            bib_score = min(weighted_score / saturation, 1.0)

            if citer_id in candidates:
                cand = candidates[citer_id]
                cand.bib_coupling_score = bib_score
                cand.bib_coupling_refs = n_refs
            elif citer_id in citer_metadata:
                meta = citer_metadata[citer_id]
                candidates[citer_id] = DiscoveryCandidate(
                    s2_id=citer_id,
                    title=meta["title"],
                    authors=meta["authors"],
                    year=meta["year"],
                    doi=meta["doi"],
                    abstract=meta["abstract"],
                    bib_coupling_score=bib_score,
                    bib_coupling_refs=n_refs,
                    pdf_url=meta.get("pdf_url"),
                    venue=meta.get("venue"),
                )

    def _expand_bibcoupling_local(
        self,
        ref_graph: dict[str, set[str]],
        library_s2_ids: set[str],
        candidates: dict[str, DiscoveryCandidate],
    ) -> None:
        """Bibliographic coupling via local graph + batch metadata fetch.

        With local lookups we use ALL refs from ref_graph (no max_core_refs limit),
        making bib coupling much more comprehensive.
        """
        assert self.graph is not None

        # Use all refs, prioritized by library overlap count
        all_refs = sorted(ref_graph.items(), key=lambda x: len(x[1]), reverse=True)
        ref_weights = {rid: len(lids) for rid, lids in all_refs}
        total_weight = sum(ref_weights.values())

        citer_weighted_scores: dict[str, float] = defaultdict(float)
        citer_ref_sets: dict[str, set[str]] = defaultdict(set)

        for i, (ref_id, _lib_ids) in enumerate(all_refs):
            weight = ref_weights[ref_id]
            if (i + 1) % 50 == 0 or i == len(all_refs) - 1:
                self._progress(
                    "bibcoupling",
                    i + 1,
                    len(all_refs),
                    f"Bib coupling (local): ref {i + 1}/{len(all_refs)}",
                )

            citer_ids = self.graph.get_citations(ref_id)

            for citer_id in citer_ids:
                if citer_id in library_s2_ids:
                    continue
                citer_ref_sets[citer_id].add(ref_id)
                citer_weighted_scores[citer_id] += weight

        # Filter to citers with >= 2 shared refs, rank by score
        saturation = max(total_weight * 0.2, 1)
        survivors: list[tuple[str, float, int]] = []

        for citer_id, weighted_score in citer_weighted_scores.items():
            n_refs = len(citer_ref_sets[citer_id])
            if n_refs < 2:
                continue
            bib_score = min(weighted_score / saturation, 1.0)
            survivors.append((citer_id, bib_score, n_refs))

        survivors.sort(key=lambda x: x[1], reverse=True)
        top_survivors = survivors[:500]

        # Batch-fetch metadata for survivors not already in candidates
        ids_needing_metadata = [s2_id for s2_id, _, _ in top_survivors if s2_id not in candidates]
        self._progress(
            "bibcoupling",
            len(all_refs),
            len(all_refs),
            f"Fetching metadata for {len(ids_needing_metadata)} bib coupling candidates...",
        )
        batch_results = (
            self.s2.get_papers_batch(ids_needing_metadata) if ids_needing_metadata else {}
        )

        # Apply scores
        for citer_id, bib_score, n_refs in top_survivors:
            if citer_id in candidates:
                cand = candidates[citer_id]
                cand.bib_coupling_score = bib_score
                cand.bib_coupling_refs = n_refs
            elif citer_id in batch_results:
                paper = batch_results[citer_id]
                candidates[citer_id] = DiscoveryCandidate(
                    s2_id=citer_id,
                    title=paper.title,
                    authors=[a for a in paper.authors if a],
                    year=paper.year,
                    doi=paper.doi,
                    abstract=paper.abstract or "",
                    bib_coupling_score=bib_score,
                    bib_coupling_refs=n_refs,
                    pdf_url=paper.pdf_url,
                    venue=paper.journal,
                )

        logger.info(
            "Bib coupling (local): %d refs searched, %d survivors, %d with metadata",
            len(all_refs),
            len(survivors),
            len([s for s in top_survivors if s[0] in candidates]),
        )

    def _compute_pagerank(
        self,
        library_s2_ids: set[str],
        candidates: dict[str, DiscoveryCandidate],
    ) -> None:
        """Compute Personalized PageRank over subgraph around library papers."""
        from incite.discovery.graph_analysis import build_subgraph, compute_pagerank
        from incite.discovery.graph_cache import GraphMetricsCache

        assert self.graph is not None

        cache = GraphMetricsCache(library_s2_ids)
        cached = cache.load()
        if cached and "pagerank" in cached:
            pagerank_scores = cached["pagerank"]
            logger.info("Using cached PageRank scores (%d entries)", len(pagerank_scores))
        else:
            int_to_s2, s2_to_int, adj = build_subgraph(self.graph, library_s2_ids, depth=2)
            seed_indices = {s2_to_int[sid] for sid in library_s2_ids if sid in s2_to_int}
            pagerank_scores = compute_pagerank(adj, seed_indices, int_to_s2)

            # Cache for reuse (co-citation may also use it)
            cocitation_scores = cached.get("cocitation", {}) if cached else {}
            cache.save(
                pagerank=pagerank_scores,
                cocitation=cocitation_scores,
                seed_count=len(library_s2_ids),
                subgraph_size=len(int_to_s2),
            )

        # Apply scores to candidates
        applied = 0
        for s2_id, cand in candidates.items():
            if s2_id in pagerank_scores:
                cand.pagerank_score = pagerank_scores[s2_id]
                applied += 1

        self._progress("pagerank", applied, applied, f"PageRank: {applied} candidates scored")
        logger.info("PageRank: scored %d/%d candidates", applied, len(candidates))

    def _compute_cocitation(
        self,
        library_s2_ids: set[str],
        candidates: dict[str, DiscoveryCandidate],
    ) -> None:
        """Compute co-citation scores with Adamic/Adar weighting."""
        from incite.discovery.graph_analysis import compute_cocitation
        from incite.discovery.graph_cache import GraphMetricsCache

        assert self.graph is not None

        cache = GraphMetricsCache(library_s2_ids)
        cached = cache.load()
        if cached and "cocitation" in cached:
            cocitation_scores = cached["cocitation"]
            logger.info("Using cached co-citation scores (%d entries)", len(cocitation_scores))
        else:
            cocitation_scores = compute_cocitation(self.graph, library_s2_ids)

            # Cache for reuse
            pagerank_scores = cached.get("pagerank", {}) if cached else {}
            cache.save(
                pagerank=pagerank_scores,
                cocitation=cocitation_scores,
                seed_count=len(library_s2_ids),
                subgraph_size=0,
            )

        # Apply scores to candidates
        applied = 0
        for s2_id, cand in candidates.items():
            if s2_id in cocitation_scores:
                cand.cocitation_score = cocitation_scores[s2_id]
                applied += 1

        self._progress("cocitation", applied, applied, f"Co-citation: {applied} candidates scored")
        logger.info("Co-citation: scored %d/%d candidates", applied, len(candidates))

    # ------------------------------------------------------------------
    # Step 4: S2 Recommendations
    # ------------------------------------------------------------------

    def _expand_recommendations(
        self,
        doi_to_s2: dict[str, str],
        papers: list[dict],
        library_s2_ids: set[str],
        candidates: dict[str, DiscoveryCandidate],
        n_recs: int = 10,
    ) -> None:
        """Use S2 Recommendations API for library papers (in-place).

        Tries the multi-paper batch endpoint first (one call for all seeds),
        falls back to per-paper calls if the batch fails.
        """
        papers_with_s2 = [
            (p, doi_to_s2[p["doi"].lower()])
            for p in papers
            if p.get("doi") and p["doi"].lower() in doi_to_s2
        ]

        # Collect all resolved S2 IDs for the batch call
        positive_ids = [s2_id for _, s2_id in papers_with_s2]
        negative_ids = list(library_s2_ids)

        if not positive_ids:
            return

        # Try batch endpoint first
        self._progress("recommendations", 0, 1, "Fetching batch recommendations...")
        rec_papers = self.s2.get_recommendations_batch(
            positive_ids, negative_ids=negative_ids, limit=500
        )

        if rec_papers:
            # Batch succeeded — score by position in ranked results
            for idx, p in enumerate(rec_papers):
                if p.id in library_s2_ids:
                    continue

                score = max(0.0, 1.0 - idx / max(len(rec_papers), 1))

                if p.id not in candidates:
                    candidates[p.id] = DiscoveryCandidate(
                        s2_id=p.id,
                        title=p.title,
                        authors=[a for a in p.authors if a],
                        year=p.year,
                        doi=p.doi,
                        abstract=p.abstract or "",
                        pdf_url=p.pdf_url,
                        venue=p.journal,
                    )

                cand = candidates[p.id]
                if score > cand.semantic_score:
                    cand.semantic_score = score
                    cand.semantic_source_title = "library"

            self._progress("recommendations", 1, 1, f"Got {len(rec_papers)} batch recs")
            return

        # Fallback: per-paper calls (original approach)
        logger.info("Batch recs failed, falling back to per-paper recommendations")
        per_paper = papers_with_s2[:n_recs]
        for i, (lib_paper, s2_id) in enumerate(per_paper):
            lib_title = lib_paper.get("title", "?")[:60]
            self._progress(
                "recommendations",
                i + 1,
                len(per_paper),
                f"Recs: {lib_title}...",
            )

            rec_papers = self.s2.get_recommendations(s2_id, limit=100)

            for idx, p in enumerate(rec_papers):
                if p.id in library_s2_ids:
                    continue

                score = max(0.0, 1.0 - idx / max(len(rec_papers), 1))

                if p.id not in candidates:
                    candidates[p.id] = DiscoveryCandidate(
                        s2_id=p.id,
                        title=p.title,
                        authors=[a for a in p.authors if a],
                        year=p.year,
                        doi=p.doi,
                        abstract=p.abstract or "",
                        pdf_url=p.pdf_url,
                        venue=p.journal,
                    )

                cand = candidates[p.id]
                if score > cand.semantic_score:
                    cand.semantic_score = score
                    cand.semantic_source_title = lib_title

    # ------------------------------------------------------------------
    # Step 5: Author expansion
    # ------------------------------------------------------------------

    def _expand_authors(
        self,
        all_papers: list[dict],
        library_s2_ids: set[str],
        candidates: dict[str, DiscoveryCandidate],
        library_titles: set[str],
    ) -> None:
        """Find recent works by prolific authors via OpenAlex (in-place)."""
        author_paper_count: dict[str, int] = Counter()
        for p in all_papers:
            for author in p.get("authors", []):
                if isinstance(author, str) and author.strip():
                    author_paper_count[author.strip()] += 1

        prolific = {a: count for a, count in author_paper_count.items() if count >= 2}
        if not prolific:
            prolific = dict(author_paper_count.most_common(10))

        sorted_authors = sorted(prolific.items(), key=lambda x: x[1], reverse=True)[:20]

        for i, (author_name, _count) in enumerate(sorted_authors):
            self._progress(
                "authors",
                i + 1,
                len(sorted_authors),
                f"Author: {author_name}",
            )

            try:
                self.oa._rate_limit()
                try:
                    resp = requests.get(
                        f"{self.oa.BASE_URL}/authors",
                        params={
                            **self.oa._params(),
                            "search": author_name,
                            "per-page": 3,
                        },
                        timeout=30,
                    )
                    resp.raise_for_status()
                    author_results = resp.json().get("results", [])
                except requests.RequestException:
                    continue

                if not author_results:
                    continue

                oa_author_id = author_results[0].get("id", "").split("/")[-1]
                if not oa_author_id:
                    continue

                self.oa._rate_limit()
                try:
                    resp = requests.get(
                        f"{self.oa.BASE_URL}/works",
                        params={
                            **self.oa._params(),
                            "filter": f"author.id:{oa_author_id},publication_year:>2020",
                            "sort": "publication_year:desc",
                            "per-page": 25,
                        },
                        timeout=30,
                    )
                    resp.raise_for_status()
                    works = resp.json().get("results", [])
                except requests.RequestException:
                    continue

                for work in works:
                    title = work.get("title", "")
                    if not title or title.lower().strip() in library_titles:
                        continue

                    raw_doi = work.get("doi", "")
                    doi = raw_doi.replace("https://doi.org/", "") if raw_doi else None

                    # Try to match existing candidate by DOI
                    matched_cand = None
                    if doi:
                        for c in candidates.values():
                            if c.doi and c.doi.lower() == doi.lower():
                                matched_cand = c
                                break

                    if matched_cand:
                        if author_name not in matched_cand.overlapping_authors:
                            matched_cand.overlapping_authors.append(author_name)
                            matched_cand.author_overlap += 1
                    else:
                        key = doi or f"oa-{hash(title)}"
                        if key not in candidates:
                            work_authors = []
                            for authorship in work.get("authorships", []):
                                a = authorship.get("author", {})
                                if a.get("display_name"):
                                    work_authors.append(a["display_name"])

                            abstract = self.oa.reconstruct_abstract(
                                work.get("abstract_inverted_index", {})
                            )

                            candidates[key] = DiscoveryCandidate(
                                s2_id=key,
                                title=title,
                                authors=work_authors,
                                year=work.get("publication_year"),
                                doi=doi,
                                abstract=abstract or "",
                                author_overlap=1,
                                overlapping_authors=[author_name],
                            )
            except Exception:
                logger.warning("Author expansion failed for %s, skipping", author_name)
                continue
