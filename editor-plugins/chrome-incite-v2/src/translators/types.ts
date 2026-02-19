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
