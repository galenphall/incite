/** A section extracted from article HTML with heading and paragraphs. */
export interface ExtractedSection {
  heading?: string;       // From <h2>, <h3>, etc.
  paragraphs: string[];   // <p> texts within this section
}

/** Structured full text extracted from article HTML. */
export interface StructuredFullText {
  sections: ExtractedSection[];
  extraction_method: string;  // "pmc", "elsevier", "generic", etc.
  source_hostname: string;    // e.g. "nature.com", "pmc.ncbi.nlm.nih.gov"
}

/** Metadata extracted from a web page about a paper. */
export interface PaperMetadata {
  title: string;
  authors?: string[];
  year?: number;
  doi?: string;
  abstract?: string;
  journal?: string;
  url?: string;
  arxiv_id?: string;
  pdf_url?: string;
  full_text?: string;  // Article body text extracted from page
  structured_text?: StructuredFullText;
}

/** Detection result from a translator. */
export interface DetectionResult {
  type: "single" | "multiple";
}

/** A translator extracts paper metadata from a specific type of web page. */
export interface Translator {
  name: string;
  urlPatterns: RegExp[];
  detect(document: Document): DetectionResult | null;
  extractSingle(document: Document): PaperMetadata | null;
  extractMultiple(document: Document): PaperMetadata[];
}
