import type { Translator, PaperMetadata, DetectionResult } from "./types";
import { extractStructuredText } from "./generic";
import { getMeta, getAllMeta, extractYear } from "./utils";

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
    const year = extractYear(getMeta(doc, "citation_date", "citation_publication_date"));
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
