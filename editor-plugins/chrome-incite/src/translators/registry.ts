import type { Translator, DetectionResult, PaperMetadata } from "./types";
import { arxivTranslator } from "./arxiv";
import { semanticScholarTranslator } from "./semantic-scholar";
import { googleScholarTranslator } from "./google-scholar";
import { biorxivTranslator } from "./biorxiv";
import { pubmedTranslator } from "./pubmed";
import { genericTranslator } from "./generic";

// Ordered by specificity -- most specific first
const TRANSLATORS: Translator[] = [
  arxivTranslator,
  semanticScholarTranslator,
  googleScholarTranslator,
  biorxivTranslator,
  pubmedTranslator,
  // generic is the fallback, not in this array
];

/** Find the best translator for a given URL. Falls back to generic. */
export function findTranslator(url: string): Translator {
  for (const t of TRANSLATORS) {
    if (t.urlPatterns.some(p => p.test(url))) {
      return t;
    }
  }
  return genericTranslator;
}

/** Check if a URL is likely an academic site (for toolbar context switching). */
export function isAcademicSite(url: string): boolean {
  // Check specific translators
  for (const t of TRANSLATORS) {
    if (t.urlPatterns.some(p => p.test(url))) return true;
  }
  // DOI resolver
  if (/doi\.org\//.test(url)) return true;
  return false;
}

// Re-export types and generic translator for content script use
export type { Translator, DetectionResult, PaperMetadata };
export { genericTranslator };
