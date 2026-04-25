/**
 * Shared meta-tag and DOM extraction utilities for translators.
 *
 * These replace per-translator copies of getMeta/getAllMeta. Uses case-insensitive
 * CSS selectors (`i` flag) for robustness — some publishers use nonstandard casing
 * like `Citation_Title`. This matches the generic translator's existing behavior.
 */

/**
 * Get the first non-empty meta tag value matching any of the given names.
 * Checks both `name` and `property` attributes (case-insensitive).
 */
export function getMeta(doc: Document, ...names: string[]): string | null {
  for (const name of names) {
    const el = doc.querySelector(`meta[name="${name}" i], meta[property="${name}" i]`);
    const content = el?.getAttribute("content")?.trim();
    if (content) return content;
  }
  return null;
}

/**
 * Get all meta tag values for a given name (case-insensitive).
 * Useful for repeated tags like `citation_author`.
 */
export function getAllMeta(doc: Document, name: string): string[] {
  return Array.from(doc.querySelectorAll(`meta[name="${name}" i]`))
    .map(el => el.getAttribute("content")?.trim())
    .filter((c): c is string => !!c);
}

/**
 * Extract a 4-digit year from a date string.
 * Handles formats like "2024", "2024/01/15", "January 2024", etc.
 */
export function extractYear(dateStr: string | null): number | undefined {
  if (!dateStr) return undefined;
  const match = dateStr.match(/(\d{4})/);
  return match ? parseInt(match[1], 10) || undefined : undefined;
}

/**
 * Get trimmed text content of the first element matching a selector.
 * Inspired by Zotero's `text()` helper.
 */
export function text(el: Element | Document, selector: string): string | null {
  const found = el.querySelector(selector);
  const content = found?.textContent?.trim();
  return content || null;
}

/**
 * Get an attribute value from the first element matching a selector.
 * Inspired by Zotero's `attr()` helper.
 */
export function attr(el: Element | Document, selector: string, attribute: string): string | null {
  const found = el.querySelector(selector);
  return found?.getAttribute(attribute)?.trim() ?? null;
}
