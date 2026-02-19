import type { Translator, PaperMetadata, DetectionResult } from "./types";

function getMeta(doc: Document, name: string): string | null {
  const el = doc.querySelector(`meta[name="${name}"], meta[property="${name}"]`);
  return el?.getAttribute("content")?.trim() ?? null;
}

/** Try to find a PDF URL from meta tags or page elements. */
function findPdfUrl(doc: Document): string | undefined {
  // citation_pdf_url meta tag (S2 pages include this)
  const metaPdf = getMeta(doc, "citation_pdf_url");
  if (metaPdf) return metaPdf;

  // S2 "View PDF" button/link
  const pdfLink = doc.querySelector<HTMLAnchorElement>(
    'a[data-heap-id="paper-link-button"][href*=".pdf"], ' +
    'a[href*="arxiv.org/pdf"], ' +
    'a.cl-paper-view-paper[href*=".pdf"]'
  );
  if (pdfLink?.href) return pdfLink.href;

  return undefined;
}

export const semanticScholarTranslator: Translator = {
  name: "semantic-scholar",
  urlPatterns: [/semanticscholar\.org\/paper\//, /semanticscholar\.org\/search/],

  detect(doc: Document): DetectionResult | null {
    const url = doc.location.href;
    if (url.includes("/search")) {
      const results = doc.querySelectorAll("[data-test-id='search-result']");
      return results.length > 0 ? { type: "multiple" } : null;
    }
    // Single paper page
    const title = getMeta(doc, "citation_title");
    return title ? { type: "single" } : null;
  },

  extractSingle(doc: Document): PaperMetadata | null {
    // Try JSON-LD first
    const jsonLd = doc.querySelector('script[type="application/ld+json"]');
    if (jsonLd) {
      try {
        const data = JSON.parse(jsonLd.textContent ?? "");
        if (data["@type"] === "ScholarlyArticle" && data.name) {
          const authors = Array.isArray(data.author)
            ? data.author.map((a: { name?: string }) => a.name).filter(Boolean)
            : undefined;
          const doi = typeof data.sameAs === "string" && data.sameAs.includes("doi.org")
            ? data.sameAs.replace(/^https?:\/\/doi\.org\//, "")
            : undefined;
          return {
            title: data.name,
            authors,
            year: data.datePublished ? parseInt(data.datePublished.match(/(\d{4})/)?.[1] ?? "", 10) || undefined : undefined,
            doi,
            abstract: data.description ?? undefined,
            journal: data.isPartOf?.name ?? undefined,
            url: doc.location.href,
            pdf_url: findPdfUrl(doc),
          };
        }
      } catch { /* fall through to meta tags */ }
    }

    // Fallback to meta tags
    const title = getMeta(doc, "citation_title");
    if (!title) return null;

    const authors = Array.from(doc.querySelectorAll('meta[name="citation_author"]'))
      .map(el => el.getAttribute("content")?.trim())
      .filter((c): c is string => !!c);

    return {
      title,
      authors: authors.length ? authors : undefined,
      doi: getMeta(doc, "citation_doi") ?? undefined,
      abstract: getMeta(doc, "og:description") ?? undefined,
      url: doc.location.href,
      pdf_url: findPdfUrl(doc),
    };
  },

  extractMultiple(doc: Document): PaperMetadata[] {
    const results: PaperMetadata[] = [];
    const cards = doc.querySelectorAll("[data-test-id='search-result']");

    for (const card of cards) {
      try {
        const titleEl = card.querySelector("h2 a, [data-test-id='title'] a");
        const title = titleEl?.textContent?.trim();
        if (!title) continue;

        // Authors
        const authorEls = card.querySelectorAll("[data-test-id='author-list'] span a, .cl-paper-authors a");
        const authors = Array.from(authorEls)
          .map(el => el.textContent?.trim())
          .filter((a): a is string => !!a);

        // Year
        const yearEl = card.querySelector("[data-test-id='paper-year'] span, .cl-paper-year");
        const yearText = yearEl?.textContent?.trim();
        const year = yearText ? parseInt(yearText.match(/(\d{4})/)?.[1] ?? "", 10) || undefined : undefined;

        // URL
        const href = titleEl?.getAttribute("href");
        const url = href?.startsWith("/") ? `https://www.semanticscholar.org${href}` : href ?? undefined;

        results.push({
          title,
          authors: authors.length ? authors : undefined,
          year,
          url,
        });
      } catch { /* skip malformed result */ }
    }

    return results;
  },
};
