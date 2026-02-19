import type { Translator, PaperMetadata, DetectionResult } from "./types";

/**
 * Extract article body text from the page DOM.
 * Tries publisher-specific selectors first, then generic article/main selectors.
 * Returns null if no substantial text found (<200 chars).
 */
export function extractFullText(doc: Document): string | null {
  // Publisher-specific selectors (ordered by specificity)
  const selectors = [
    // PMC / NCBI
    ".jig-ncbiinpagenav .tsec",
    // Elsevier / ScienceDirect
    "#body .section",
    // Nature / Springer
    "article .c-article-body",
    "article .article-body",
    "#article-body",
    // Wiley
    ".article-section__content",
    // PLOS
    ".article-content",
    // Generic semantic HTML
    '[role="main"] p',
    "article p",
    // Broadest fallback: main content paragraphs
    "main p",
  ];

  for (const selector of selectors) {
    const elements = doc.querySelectorAll(selector);
    if (elements.length === 0) continue;

    const paragraphs: string[] = [];
    for (const el of elements) {
      // For container selectors, grab paragraphs inside
      if (el.tagName !== "P") {
        const ps = el.querySelectorAll("p");
        for (const p of ps) {
          const text = p.textContent?.trim();
          if (text && text.length > 30) paragraphs.push(text);
        }
      } else {
        const text = el.textContent?.trim();
        if (text && text.length > 30) paragraphs.push(text);
      }
    }

    const fullText = paragraphs.join("\n\n");
    if (fullText.length >= 200) return fullText;
  }

  return null;
}

function getMeta(doc: Document, names: string[]): string | null {
  for (const name of names) {
    const el = doc.querySelector(`meta[name="${name}" i], meta[property="${name}" i]`);
    if (el) {
      const content = el.getAttribute("content");
      if (content?.trim()) return content.trim();
    }
  }
  return null;
}

function getAllMeta(doc: Document, name: string): string[] {
  const els = doc.querySelectorAll(`meta[name="${name}" i], meta[property="${name}" i]`);
  return Array.from(els)
    .map(el => el.getAttribute("content")?.trim())
    .filter((c): c is string => !!c);
}

function parseYear(dateStr: string | null): number | undefined {
  if (!dateStr) return undefined;
  const match = dateStr.match(/(\d{4})/);
  return match ? parseInt(match[1], 10) : undefined;
}

export const genericTranslator: Translator = {
  name: "generic",
  urlPatterns: [], // Used as fallback, doesn't match by URL

  detect(doc: Document): DetectionResult | null {
    const title = getMeta(doc, ["citation_title", "DC.Title", "DC.title"]);
    return title ? { type: "single" } : null;
  },

  extractSingle(doc: Document): PaperMetadata | null {
    const title = getMeta(doc, ["citation_title", "DC.Title", "DC.title", "og:title"]);
    if (!title) return null;

    const authors = getAllMeta(doc, "citation_author");
    if (!authors.length) {
      const dcAuthors = getAllMeta(doc, "DC.Creator");
      if (dcAuthors.length) authors.push(...dcAuthors);
    }

    const doi = getMeta(doc, ["citation_doi", "DC.Identifier"]) ?? undefined;
    const abstract = getMeta(doc, ["citation_abstract", "DC.Description", "og:description"]) ?? undefined;
    const journal = getMeta(doc, ["citation_journal_title", "DC.Source"]) ?? undefined;
    const year = parseYear(getMeta(doc, ["citation_date", "citation_publication_date", "DC.Date"]));
    const pdf_url = getMeta(doc, ["citation_pdf_url"]) ?? undefined;

    const full_text = extractFullText(doc) ?? undefined;

    return {
      title,
      authors: authors.length ? authors : undefined,
      year,
      doi,
      abstract,
      journal,
      url: doc.location.href,
      pdf_url,
      full_text,
    };
  },

  extractMultiple(_doc: Document): PaperMetadata[] {
    // Generic translator only handles single papers
    return [];
  },
};
