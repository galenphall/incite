import type { Translator, PaperMetadata, DetectionResult, ExtractedSection, StructuredFullText } from "./types";

/** Selectors for noise elements to remove before extraction. */
const NOISE_SELECTORS = [
  "nav", "footer", ".cookie-consent", ".cookie-banner", ".cookie-notice",
  ".share-tools", ".share-widget", ".social-share",
  ".author-info", ".author-notes",
  ".metrics", ".altmetric",
  ".supplementary-data", ".supplementary-materials",
  '[role="navigation"]', '[role="banner"]',
  ".sidebar", "#sidebar",
  ".advertisement", ".ad-container",
  ".related-articles", ".recommended-articles",
  ".references", "#references",
  ".footnotes", ".endnotes",
  // ScienceDirect / Elsevier
  ".bibliography", ".Footnotes", ".Tail",
  ".RelatedContent", ".ReferencedArticles",
  ".ListArticles", ".Copyright",
  "figure", ".figure", ".table-wrap",
].join(", ");

/** Block-level tags that disqualify a div from being treated as a paragraph. */
const BLOCK_TAGS = new Set([
  "div", "section", "article", "aside", "blockquote",
  "table", "figure", "ul", "ol", "pre", "form",
  "header", "footer", "nav", "main",
]);

/** Inline citation markers to strip: [1], [1,2], [1-3], superscript digits. */
const CITATION_MARKER_RE = /\[(\d+(?:[,\s]*\d+)*(?:\s*[-–]\s*\d+)?)\]/g;
const SUPERSCRIPT_DIGITS_RE = /[⁰¹²³⁴⁵⁶⁷⁸⁹]+/g;

function cleanParagraphText(text: string): string {
  return text
    .replace(CITATION_MARKER_RE, "")
    .replace(SUPERSCRIPT_DIGITS_RE, "")
    .replace(/[\u00a0\u200b\u200c\u200d\ufeff]/g, " ")
    .replace(/ ([.,;:!?])/g, "$1")
    .replace(/  +/g, " ")
    .trim();
}

/**
 * Check if an element is a "leaf" text block — contains substantial text
 * with only inline children (no nested divs, sections, tables, etc.).
 */
function isLeafTextBlock(el: Element): boolean {
  for (const child of el.children) {
    if (BLOCK_TAGS.has(child.tagName.toLowerCase())) {
      return false;
    }
  }
  return true;
}

/**
 * Extract structured text with section headings from the page DOM.
 * Returns both flat text (backward-compat) and structured sections.
 */
export function extractStructuredText(doc: Document): { full_text: string | null; structured_text: StructuredFullText | null } {
  // Publisher-specific container selectors (ordered by specificity)
  const containerSelectors = [
    ".jig-ncbiinpagenav .tsec",
    "#body .section",
    "article .c-article-body",
    "article .article-body",
    "#article-body",
    ".article-section__content",
    ".article-content",
    ".Body",                    // ScienceDirect / Elsevier
    '[role="main"]',
    "article",
    "main",
  ];

  // Determine extraction method from hostname
  const hostname = doc.location?.hostname ?? "";
  let extractionMethod = "generic";
  if (hostname.includes("ncbi.nlm.nih.gov")) extractionMethod = "pmc";
  else if (hostname.includes("sciencedirect.com")) extractionMethod = "elsevier";
  else if (hostname.includes("nature.com") || hostname.includes("springer.com")) extractionMethod = "springer";
  else if (hostname.includes("wiley.com")) extractionMethod = "wiley";
  else if (hostname.includes("plos.org")) extractionMethod = "plos";
  else if (hostname.includes("arxiv.org")) extractionMethod = "arxiv";
  else if (hostname.includes("biorxiv.org") || hostname.includes("medrxiv.org")) extractionMethod = "biorxiv";

  for (const containerSelector of containerSelectors) {
    const containers = doc.querySelectorAll(containerSelector);
    if (containers.length === 0) continue;

    // Clone and clean the containers to avoid modifying the live DOM
    const wrapper = doc.createElement("div");
    for (const container of containers) {
      wrapper.appendChild(container.cloneNode(true));
    }

    // Remove noise elements
    const noiseEls = wrapper.querySelectorAll(NOISE_SELECTORS);
    for (const el of noiseEls) {
      el.remove();
    }

    // Walk children to extract sections with headings.
    // Match <p> and <div> elements that are leaf text blocks (no block children).
    const sections: ExtractedSection[] = [];
    let currentSection: ExtractedSection = { paragraphs: [] };

    const elements = wrapper.querySelectorAll("h2, h3, h4, p, div");
    for (const el of elements) {
      const tagName = el.tagName.toLowerCase();

      if (tagName === "h2" || tagName === "h3" || tagName === "h4") {
        // Start a new section if current has content
        if (currentSection.paragraphs.length > 0) {
          sections.push(currentSection);
        }
        const headingText = el.textContent?.trim() ?? "";
        currentSection = { heading: headingText || undefined, paragraphs: [] };
      } else {
        // For <div> elements, only treat as paragraph if it's a leaf text block
        if (tagName === "div" && !isLeafTextBlock(el)) continue;

        const rawText = el.textContent?.trim();
        if (rawText && rawText.length > 30) {
          const cleaned = cleanParagraphText(rawText);
          if (cleaned.length > 30) {
            currentSection.paragraphs.push(cleaned);
          }
        }
      }
    }

    // Push final section
    if (currentSection.paragraphs.length > 0) {
      sections.push(currentSection);
    }

    // Build flat text for backward compat
    const allParagraphs: string[] = [];
    for (const section of sections) {
      for (const para of section.paragraphs) {
        allParagraphs.push(para);
      }
    }
    const fullText = allParagraphs.join("\n\n");

    if (fullText.length >= 200) {
      return {
        full_text: fullText,
        structured_text: {
          sections,
          extraction_method: extractionMethod,
          source_hostname: hostname,
        },
      };
    }
  }

  return { full_text: null, structured_text: null };
}

/**
 * Extract article body text from the page DOM.
 * Backward-compatible wrapper around extractStructuredText().
 * Returns null if no substantial text found (<200 chars).
 */
export function extractFullText(doc: Document): string | null {
  return extractStructuredText(doc).full_text;
}

/**
 * Extract abstract from DOM elements when meta tags are missing or truncated.
 * Tries common abstract container selectors used by major publishers.
 */
function extractAbstractFromDom(doc: Document): string | null {
  const selectors = [
    ".Abstracts .abstract.author",   // ScienceDirect / Elsevier
    ".abstract-content",             // Various publishers
    '[class*="abstract"] p',         // Generic
    "#abstract p",                   // Common ID-based
    ".hlFld-Abstract p",             // Taylor & Francis
    ".abstractSection",              // Wiley
  ];

  for (const selector of selectors) {
    const els = doc.querySelectorAll(selector);
    if (els.length === 0) continue;

    const texts: string[] = [];
    for (const el of els) {
      const text = el.textContent?.trim();
      if (text && text.length > 30) {
        texts.push(cleanParagraphText(text));
      }
    }
    const combined = texts.join(" ");
    if (combined.length > 100) {
      // Strip leading "Abstract" label if present
      return combined.replace(/^Abstract\s*/i, "");
    }
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
    const journal = getMeta(doc, ["citation_journal_title", "DC.Source"]) ?? undefined;
    const year = parseYear(getMeta(doc, ["citation_date", "citation_publication_date", "DC.Date"]));
    const pdf_url = getMeta(doc, ["citation_pdf_url"]) ?? undefined;

    // Abstract: prefer meta tags, but fall back to DOM extraction if truncated
    let abstract = getMeta(doc, ["citation_abstract", "DC.Description", "og:description"]) ?? undefined;
    if (!abstract || abstract.length < 200) {
      const domAbstract = extractAbstractFromDom(doc);
      if (domAbstract && domAbstract.length > (abstract?.length ?? 0)) {
        abstract = domAbstract;
      }
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
    // Generic translator only handles single papers
    return [];
  },
};
