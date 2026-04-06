import type { Translator, PaperMetadata, DetectionResult } from "./types";
import { extractStructuredText } from "./generic";

function getMeta(doc: Document, name: string): string | null {
  const el = doc.querySelector(`meta[name="${name}"], meta[property="${name}"]`);
  return el?.getAttribute("content")?.trim() ?? null;
}

function getAllMeta(doc: Document, name: string): string[] {
  return Array.from(doc.querySelectorAll(`meta[name="${name}"]`))
    .map(el => el.getAttribute("content")?.trim())
    .filter((c): c is string => !!c);
}

export const biorxivTranslator: Translator = {
  name: "biorxiv",
  urlPatterns: [/biorxiv\.org\/content\//, /medrxiv\.org\/content\//],

  detect(doc: Document): DetectionResult | null {
    const title = getMeta(doc, "citation_title");
    return title ? { type: "single" } : null;
  },

  extractSingle(doc: Document): PaperMetadata | null {
    const title = getMeta(doc, "citation_title");
    if (!title) return null;

    const authors = getAllMeta(doc, "citation_author");
    const doi = getMeta(doc, "citation_doi") ?? undefined;
    const journal = getMeta(doc, "citation_journal_title") ?? undefined;
    const dateStr = getMeta(doc, "citation_date") ?? getMeta(doc, "citation_publication_date");
    const year = dateStr ? parseInt(dateStr.match(/(\d{4})/)?.[1] ?? "", 10) || undefined : undefined;
    const pdf_url = getMeta(doc, "citation_pdf_url") ?? undefined;

    // bioRxiv/medRxiv abstract is in the page content
    let abstract: string | undefined;
    const abstractDiv = doc.querySelector(".abstract, #abstract");
    if (abstractDiv) {
      abstract = abstractDiv.textContent?.replace(/^Abstract\s*/i, "").trim();
    }

    const { full_text, structured_text } = extractStructuredText(doc);

    return {
      title,
      authors: authors.length ? authors : undefined,
      year,
      doi,
      abstract,
      journal,
      url: doc.location.href,
      pdf_url,
      full_text: full_text ?? undefined,
      structured_text: structured_text ?? undefined,
    };
  },

  extractMultiple(_doc: Document): PaperMetadata[] {
    return [];
  },
};
