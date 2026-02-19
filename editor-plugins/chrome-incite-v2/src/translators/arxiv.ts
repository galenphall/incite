import type { Translator, PaperMetadata, DetectionResult } from "./types";
import { extractFullText } from "./generic";

function getMeta(doc: Document, name: string): string | null {
  const el = doc.querySelector(`meta[name="${name}"], meta[property="${name}"]`);
  return el?.getAttribute("content")?.trim() ?? null;
}

function getAllMeta(doc: Document, name: string): string[] {
  return Array.from(doc.querySelectorAll(`meta[name="${name}"]`))
    .map(el => el.getAttribute("content")?.trim())
    .filter((c): c is string => !!c);
}

function extractArxivId(url: string): string | null {
  // Matches: arxiv.org/abs/1706.03762 or arxiv.org/abs/2301.12345v2 or arxiv.org/pdf/1706.03762 or arxiv.org/html/1706.03762
  const match = url.match(/arxiv\.org\/(?:abs|pdf|html)\/(\d{4}\.\d{4,5}(?:v\d+)?)/);
  return match ? match[1] : null;
}

export const arxivTranslator: Translator = {
  name: "arxiv",
  urlPatterns: [/arxiv\.org\/(?:abs|pdf|html)\//],

  detect(doc: Document): DetectionResult | null {
    const arxivId = extractArxivId(doc.location.href);
    const title = getMeta(doc, "citation_title");
    return (arxivId || title) ? { type: "single" } : null;
  },

  extractSingle(doc: Document): PaperMetadata | null {
    const title = getMeta(doc, "citation_title");
    if (!title) return null;

    const authors = getAllMeta(doc, "citation_author");
    const doi = getMeta(doc, "citation_doi") ?? undefined;
    const arxivId = extractArxivId(doc.location.href) ?? undefined;
    const dateStr = getMeta(doc, "citation_date");
    const year = dateStr ? parseInt(dateStr.match(/(\d{4})/)?.[1] ?? "", 10) || undefined : undefined;
    const pdf_url = getMeta(doc, "citation_pdf_url") ?? undefined;

    // arXiv abstracts are in the page content, not meta tags
    let abstract: string | undefined;
    const abstractBlock = doc.querySelector(".abstract");
    if (abstractBlock) {
      abstract = abstractBlock.textContent?.replace(/^Abstract:\s*/i, "").trim();
    }

    // On arxiv HTML pages (/html/), extract full text; on abs pages, skip (only abstract)
    const isHtmlPage = doc.location.href.includes("/html/");
    const full_text = isHtmlPage ? (extractFullText(doc) ?? undefined) : undefined;

    return {
      title,
      authors: authors.length ? authors : undefined,
      year,
      doi,
      abstract,
      journal: "arXiv",
      url: doc.location.href,
      arxiv_id: arxivId,
      pdf_url,
      full_text,
    };
  },

  extractMultiple(_doc: Document): PaperMetadata[] {
    return [];
  },
};
