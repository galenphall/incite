/**
 * Resolves cursor context (extracted from the Google Docs texteventtarget iframe)
 * to a precise zero-based UTF-16 document index for insertion.
 *
 * Two-step process:
 * 1. Content script probes `.docs-texteventtarget-iframe` to extract text around cursor
 * 2. This resolver maps that extracted context to a document index using the REST API document
 */

import type { DocsDocument, DocsStructuralElement, DocsParagraph } from "./gdocs-api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CursorContext {
  /** Text of the paragraph the cursor is in (from texteventtarget probing) */
  paragraphText: string;
  /** Character offset within the paragraph where cursor is */
  cursorOffset: number;
  /** Text immediately before cursor (from probing left) — up to ~50 chars */
  textBefore?: string;
  /** Text immediately after cursor (from probing right) — up to ~50 chars */
  textAfter?: string;
}

export interface ResolvedPosition {
  /** Zero-based document index for the cursor position */
  index: number;
  /** Confidence: "exact" if unique match, "best-guess" if disambiguated */
  confidence: "exact" | "best-guess";
  /** The full paragraph text at this position (for verification) */
  paragraphText: string;
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

interface DocumentParagraph {
  text: string;
  startIndex: number;
  endIndex: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Normalize whitespace for fuzzy comparison. */
function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

/** Extract the plain text content of a paragraph element. */
function paragraphToText(paragraph: DocsParagraph): string {
  return paragraph.elements
    .map((e) => e.textRun?.content ?? "")
    .join("")
    .replace(/\n$/, ""); // Remove trailing newline that Docs adds
}

/** Extract all paragraphs from the document with their index ranges. */
function extractParagraphs(doc: DocsDocument): DocumentParagraph[] {
  const paragraphs: DocumentParagraph[] = [];

  function processElements(elements: DocsStructuralElement[]): void {
    for (const el of elements) {
      if (el.paragraph) {
        const text = paragraphToText(el.paragraph);
        if (text.length > 0) {
          paragraphs.push({ text, startIndex: el.startIndex, endIndex: el.endIndex });
        }
      }
      if (el.table) {
        for (const row of el.table.tableRows) {
          for (const cell of row.tableCells) {
            processElements(cell.content);
          }
        }
      }
    }
  }

  processElements(doc.body.content);
  return paragraphs;
}

/**
 * Score how well a candidate paragraph matches the cursor's textBefore/textAfter
 * context at a given offset. Higher is better.
 */
function scoreContextMatch(
  paragraphs: DocumentParagraph[],
  candidate: DocumentParagraph,
  cursorOffset: number,
  textBefore?: string,
  textAfter?: string
): number {
  let score = 0;

  // Build a window of surrounding text from the full document for cross-paragraph matching
  const docIndex = candidate.startIndex + cursorOffset;

  if (textBefore) {
    const beforeStart = docIndex - textBefore.length;
    if (beforeStart >= 0) {
      // Reconstruct the text before the cursor from paragraphs
      const reconstructed = reconstructTextRange(paragraphs, beforeStart, docIndex);
      const normalizedReconstructed = normalizeWhitespace(reconstructed);
      const normalizedBefore = normalizeWhitespace(textBefore);
      if (normalizedReconstructed === normalizedBefore) {
        score += 2;
      } else if (normalizedReconstructed.endsWith(normalizedBefore.slice(-20))) {
        score += 1;
      }
    }
  }

  if (textAfter) {
    const afterEnd = docIndex + textAfter.length;
    const reconstructed = reconstructTextRange(paragraphs, docIndex, afterEnd);
    const normalizedReconstructed = normalizeWhitespace(reconstructed);
    const normalizedAfter = normalizeWhitespace(textAfter);
    if (normalizedReconstructed === normalizedAfter) {
      score += 2;
    } else if (normalizedReconstructed.startsWith(normalizedAfter.slice(0, 20))) {
      score += 1;
    }
  }

  return score;
}

/**
 * Reconstruct text from the document's paragraph list for a given index range.
 * This is approximate — it concatenates paragraph texts with newline separators.
 */
function reconstructTextRange(
  paragraphs: DocumentParagraph[],
  startIdx: number,
  endIdx: number
): string {
  let result = "";
  for (const para of paragraphs) {
    if (para.endIndex <= startIdx) continue;
    if (para.startIndex >= endIdx) break;

    const clipStart = Math.max(0, startIdx - para.startIndex);
    const clipEnd = Math.min(para.text.length, endIdx - para.startIndex);
    if (clipStart < clipEnd && clipStart < para.text.length) {
      result += para.text.slice(clipStart, clipEnd);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Main resolver
// ---------------------------------------------------------------------------

/**
 * Resolve cursor context to a document index.
 *
 * Strategy:
 * 1. Extract all paragraph texts from the document with their index ranges
 * 2. Find paragraphs that contain the cursor's paragraphText (fuzzy match —
 *    the texteventtarget text may have minor whitespace differences)
 * 3. If unique match: use cursorOffset within that paragraph -> exact index
 * 4. If multiple matches: disambiguate using textBefore/textAfter
 * 5. If no match: try substring matching with progressively shorter text
 * 6. If still no match: return null
 */
export function resolveDocumentIndex(
  doc: DocsDocument,
  cursor: CursorContext
): ResolvedPosition | null {
  const paragraphs = extractParagraphs(doc);
  if (paragraphs.length === 0) return null;

  const normalizedCursor = normalizeWhitespace(cursor.paragraphText);
  if (normalizedCursor.length === 0) return null;

  // Step 1: Find exact normalized matches
  const exactMatches = paragraphs.filter(
    (p) => normalizeWhitespace(p.text) === normalizedCursor
  );

  if (exactMatches.length === 1) {
    const para = exactMatches[0];
    const index = para.startIndex + clampOffset(cursor.cursorOffset, para.text.length);
    return { index, confidence: "exact", paragraphText: para.text };
  }

  if (exactMatches.length > 1) {
    const best = disambiguate(paragraphs, exactMatches, cursor);
    if (best) return best;
  }

  // Step 2: Try contains match (cursor text is a substring of paragraph or vice versa)
  const containsMatches = paragraphs.filter((p) => {
    const normalizedPara = normalizeWhitespace(p.text);
    return normalizedPara.includes(normalizedCursor) || normalizedCursor.includes(normalizedPara);
  });

  if (containsMatches.length === 1) {
    const para = containsMatches[0];
    const index = para.startIndex + clampOffset(cursor.cursorOffset, para.text.length);
    return { index, confidence: "best-guess", paragraphText: para.text };
  }

  if (containsMatches.length > 1) {
    const best = disambiguate(paragraphs, containsMatches, cursor);
    if (best) return best;
  }

  // Step 3: Progressive substring matching — drop words from start and end
  const words = normalizedCursor.split(" ");
  const minWords = Math.max(3, Math.floor(words.length / 3));

  for (let drop = 1; drop <= words.length - minWords; drop++) {
    // Drop from end
    const endTrimmed = words.slice(0, words.length - drop).join(" ");
    const endMatches = paragraphs.filter((p) =>
      normalizeWhitespace(p.text).includes(endTrimmed)
    );
    if (endMatches.length === 1) {
      const para = endMatches[0];
      const index = para.startIndex + clampOffset(cursor.cursorOffset, para.text.length);
      return { index, confidence: "best-guess", paragraphText: para.text };
    }

    // Drop from start
    const startTrimmed = words.slice(drop).join(" ");
    const startMatches = paragraphs.filter((p) =>
      normalizeWhitespace(p.text).includes(startTrimmed)
    );
    if (startMatches.length === 1) {
      const para = startMatches[0];
      // Adjust cursor offset: we dropped words from the start, so we need to
      // find where the trimmed text begins in the paragraph
      const offsetInPara = normalizeWhitespace(para.text).indexOf(startTrimmed);
      const adjustedOffset = offsetInPara + Math.max(0, cursor.cursorOffset - substringLength(words, 0, drop));
      const index = para.startIndex + clampOffset(adjustedOffset, para.text.length);
      return { index, confidence: "best-guess", paragraphText: para.text };
    }
  }

  // No match found
  return null;
}

/** Clamp an offset to be within [0, maxLen]. */
function clampOffset(offset: number, maxLen: number): number {
  return Math.max(0, Math.min(offset, maxLen));
}

/** Calculate the character length of a range of words (with spaces). */
function substringLength(words: string[], startWord: number, endWord: number): number {
  if (startWord >= endWord || startWord >= words.length) return 0;
  const slice = words.slice(startWord, endWord);
  return slice.join(" ").length + 1; // +1 for the trailing space
}

/**
 * Disambiguate among multiple matching paragraphs using textBefore/textAfter.
 * Returns the best match or null if disambiguation fails.
 */
function disambiguate(
  allParagraphs: DocumentParagraph[],
  candidates: DocumentParagraph[],
  cursor: CursorContext
): ResolvedPosition | null {
  if (!cursor.textBefore && !cursor.textAfter) return null;

  let bestScore = -1;
  let bestCandidate: DocumentParagraph | null = null;

  for (const candidate of candidates) {
    const offset = clampOffset(cursor.cursorOffset, candidate.text.length);
    const score = scoreContextMatch(
      allParagraphs,
      candidate,
      offset,
      cursor.textBefore,
      cursor.textAfter
    );
    if (score > bestScore) {
      bestScore = score;
      bestCandidate = candidate;
    }
  }

  if (bestCandidate && bestScore > 0) {
    const offset = clampOffset(cursor.cursorOffset, bestCandidate.text.length);
    return {
      index: bestCandidate.startIndex + offset,
      confidence: "best-guess",
      paragraphText: bestCandidate.text,
    };
  }

  return null;
}

// ---------------------------------------------------------------------------
// Exported utilities
// ---------------------------------------------------------------------------

/** Extract full plain text from the document for context extraction. */
export function extractFullText(doc: DocsDocument): string {
  const paragraphs = extractParagraphs(doc);
  return paragraphs.map((p) => p.text).join("\n");
}

/** Get the document index corresponding to a character offset in the full text. */
export function fullTextOffsetToDocIndex(
  doc: DocsDocument,
  offset: number
): number | null {
  const paragraphs = extractParagraphs(doc);
  let currentOffset = 0;

  for (const para of paragraphs) {
    const paraLen = para.text.length + 1; // +1 for the \n join separator
    if (offset < currentOffset + para.text.length) {
      const withinPara = offset - currentOffset;
      return para.startIndex + withinPara;
    }
    currentOffset += paraLen;
  }

  return null;
}
