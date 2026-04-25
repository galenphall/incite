/**
 * Citation and bibliography insertion into Google Docs via the REST API.
 *
 * Uses named ranges (INCITE_CIT_<paperId>_<uuid8>, INCITE_BIB) to track citations
 * and bibliography sections so they can be reformatted, scanned, or cleaned.
 * Each citation instance gets a unique 8-hex-char suffix so the same paper
 * can be cited multiple times without duplicate range names.
 */

import type {
  GoogleDocsAPI,
  DocsRequest,
  DocsNamedRange,
  DocsDocument,
} from "./gdocs-api";
import type { TrackedCitation } from "@incite/shared";

/** Named range prefix for inCite citations. */
export const CITATION_RANGE_PREFIX = "INCITE_CIT_";

/** Named range name for the bibliography section. */
export const BIBLIOGRAPHY_RANGE_NAME = "INCITE_BIB";

/** Generate an 8-character hex UUID for instance-unique naming. */
export function generateUuid8(): string {
  const bytes = new Uint8Array(4);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Parsed result of a citation named range name. */
export interface ParsedCitationRange {
  paperId: string;
  instanceId: string | null; // null for legacy ranges without uuid8
}

/** Parse a named range name into paperId and optional instanceId. */
export function parseCitationRangeName(
  name: string
): ParsedCitationRange | null {
  if (!name.startsWith(CITATION_RANGE_PREFIX)) return null;
  const suffix = name.slice(CITATION_RANGE_PREFIX.length);
  const lastUnderscore = suffix.lastIndexOf("_");
  if (lastUnderscore > 0) {
    const tail = suffix.slice(lastUnderscore + 1);
    if (/^[0-9a-f]{8}$/.test(tail)) {
      return { paperId: suffix.slice(0, lastUnderscore), instanceId: tail };
    }
  }
  return { paperId: suffix, instanceId: null }; // legacy format
}

/** Build a named range name for a new citation instance. */
export function buildCitationRangeName(paperId: string): string {
  return `${CITATION_RANGE_PREFIX}${paperId}_${generateUuid8()}`;
}

/** A citation instance found in the document via named ranges. */
export interface CitationInstance {
  paperId: string;
  instanceId: string | null;
  rangeName: string;
  namedRangeId: string;
  startIndex: number;
  endIndex: number;
}

/** Result of an enhanced scan of the document for citation named ranges. */
export interface EnhancedScanResult {
  /** All citation instances found, keyed by paperId. */
  instances: Map<string, CitationInstance[]>;
  /** Paper IDs found in doc but not in tracker. */
  untracked: string[];
  /** Paper IDs in tracker but not found in doc. */
  orphaned: string[];
  /** Range names that appear more than once (copy-paste duplicates). */
  duplicateRanges: string[];
}

export class GDocsCitationInserter {
  constructor(
    private api: GoogleDocsAPI,
    private docId: string
  ) {}

  /**
   * Insert a single citation at the given document index.
   * Creates a batchUpdate with: insert text, add hyperlink, create named range.
   * Returns the new end index (for chaining insertions).
   */
  async insertCitation(
    index: number,
    text: string,
    paperUrl: string,
    paperId: string
  ): Promise<{ endIndex: number }> {
    const requests: DocsRequest[] = [
      { insertText: { location: { index }, text } },
      {
        updateTextStyle: {
          range: { startIndex: index, endIndex: index + text.length },
          textStyle: { link: { url: paperUrl } },
          fields: "link",
        },
      },
      {
        createNamedRange: {
          name: buildCitationRangeName(paperId),
          range: { startIndex: index, endIndex: index + text.length },
        },
      },
    ];

    await this.api.batchUpdate(this.docId, requests);
    return { endIndex: index + text.length };
  }

  /**
   * Insert a grouped citation (multiple papers) at the given index.
   * Each segment gets its own hyperlink and named range.
   * The segments array defines the individual citations within the group.
   *
   * Example: "(Hall et al., 2024; Smith, 2023)" where each author-year
   * is a separate segment with its own link.
   */
  async insertGroupedCitation(
    index: number,
    fullText: string,
    segments: {
      text: string;
      paperUrl: string;
      paperId: string;
      offsetInFullText: number;
    }[]
  ): Promise<{ endIndex: number }> {
    const requests: DocsRequest[] = [
      // First: insert the full text
      { insertText: { location: { index }, text: fullText } },
    ];

    // Then: add hyperlinks and named ranges for each segment
    for (const seg of segments) {
      const segStart = index + seg.offsetInFullText;
      const segEnd = segStart + seg.text.length;
      requests.push({
        updateTextStyle: {
          range: { startIndex: segStart, endIndex: segEnd },
          textStyle: { link: { url: seg.paperUrl } },
          fields: "link",
        },
      });
      requests.push({
        createNamedRange: {
          name: buildCitationRangeName(seg.paperId),
          range: { startIndex: segStart, endIndex: segEnd },
        },
      });
    }

    await this.api.batchUpdate(this.docId, requests);
    return { endIndex: index + fullText.length };
  }

  /**
   * Insert or update a bibliography section at the end of the document.
   *
   * If an existing INCITE_BIB named range exists, replaces its content.
   * Otherwise, appends at the end of the document.
   */
  async insertBibliography(
    entries: { paperId: string; formatted: string; url?: string }[]
  ): Promise<void> {
    // Get the document to find existing bibliography and document length
    const doc = await this.api.getDocument(this.docId);
    const existingRanges = this.findBibliographyRange(doc);

    const header = "\n\nReferences\n";
    const body = entries.map((e) => e.formatted).join("\n") + "\n";
    const fullText = header + body;

    if (existingRanges.length > 0) {
      // Replace existing bibliography: delete old content, insert new
      // Use the first range (there should only be one)
      const range = existingRanges[0];
      const oldStart = range.ranges[0].startIndex;
      const oldEnd = range.ranges[range.ranges.length - 1].endIndex;

      const requests: DocsRequest[] = [
        // Delete the named range first
        { deleteNamedRange: { namedRangeId: range.namedRangeId } },
        // Delete old content
        {
          deleteContentRange: {
            range: { startIndex: oldStart, endIndex: oldEnd },
          },
        },
        // Insert new content at the same position
        { insertText: { location: { index: oldStart }, text: fullText } },
        // Make "References" bold
        {
          updateTextStyle: {
            range: {
              startIndex: oldStart + 2,
              endIndex: oldStart + 2 + "References".length,
            },
            textStyle: { bold: true },
            fields: "bold",
          },
        },
        // Create new named range for the whole bibliography
        {
          createNamedRange: {
            name: BIBLIOGRAPHY_RANGE_NAME,
            range: {
              startIndex: oldStart,
              endIndex: oldStart + fullText.length,
            },
          },
        },
      ];

      await this.api.batchUpdate(this.docId, requests);
    } else {
      // Append at end of document
      const bodyContent = doc.body.content;
      const lastElement = bodyContent[bodyContent.length - 1];
      const endIndex = lastElement.endIndex - 1; // -1 for final newline

      const requests: DocsRequest[] = [
        { insertText: { location: { index: endIndex }, text: fullText } },
        // Make "References" bold (account for the leading \n\n)
        {
          updateTextStyle: {
            range: {
              startIndex: endIndex + 2,
              endIndex: endIndex + 2 + "References".length,
            },
            textStyle: { bold: true },
            fields: "bold",
          },
        },
        {
          createNamedRange: {
            name: BIBLIOGRAPHY_RANGE_NAME,
            range: {
              startIndex: endIndex,
              endIndex: endIndex + fullText.length,
            },
          },
        },
      ];

      await this.api.batchUpdate(this.docId, requests);
    }
  }

  /**
   * Scan the document for all INCITE_CIT_* named ranges.
   * Returns found paper IDs and any orphaned ones (tracked but not in doc).
   */
  async scanCitations(
    trackedPaperIds: string[]
  ): Promise<{ found: string[]; orphaned: string[] }> {
    const doc = await this.api.getDocument(this.docId);
    const found = new Set<string>();

    if (doc.namedRanges) {
      for (const name of Object.keys(doc.namedRanges)) {
        const parsed = parseCitationRangeName(name);
        if (parsed) {
          found.add(parsed.paperId);
        }
      }
    }

    const orphaned = trackedPaperIds.filter((id) => !found.has(id));
    return { found: Array.from(found), orphaned };
  }

  /**
   * Enhanced scan: returns per-instance data and detects copy-paste duplicates.
   * Unlike scanCitations(), this returns full CitationInstance objects and
   * identifies ranges that share the same name (copy-paste artifacts).
   */
  async scanCitationsEnhanced(
    trackedPaperIds: string[]
  ): Promise<EnhancedScanResult> {
    const doc = await this.api.getDocument(this.docId);
    const instances = new Map<string, CitationInstance[]>();
    const rangeNameCounts = new Map<string, number>();

    if (doc.namedRanges) {
      for (const [rangeName, entry] of Object.entries(doc.namedRanges)) {
        const parsed = parseCitationRangeName(rangeName);
        if (!parsed) continue;

        const ranges = entry.namedRanges ?? [];
        // Track how many named range objects share this name
        rangeNameCounts.set(rangeName, ranges.length);

        for (const nr of ranges) {
          const rangeInfo = nr.ranges?.[0];
          if (!rangeInfo) continue;

          const instance: CitationInstance = {
            paperId: parsed.paperId,
            instanceId: parsed.instanceId,
            rangeName,
            namedRangeId: nr.namedRangeId,
            startIndex: rangeInfo.startIndex ?? 0,
            endIndex: rangeInfo.endIndex ?? 0,
          };

          const existing = instances.get(parsed.paperId) ?? [];
          existing.push(instance);
          instances.set(parsed.paperId, existing);
        }
      }
    }

    const foundPaperIds = new Set(instances.keys());
    const trackedSet = new Set(trackedPaperIds);

    const untracked = Array.from(foundPaperIds).filter((id) => !trackedSet.has(id));
    const orphaned = trackedPaperIds.filter((id) => !foundPaperIds.has(id));

    // Duplicate ranges: same range name appears more than once (copy-paste artifact)
    const duplicateRanges: string[] = [];
    for (const [rangeName, count] of rangeNameCounts) {
      if (count > 1) {
        duplicateRanges.push(rangeName);
      }
    }

    return { instances, untracked, orphaned, duplicateRanges };
  }

  /**
   * Fix duplicate named ranges caused by copy-paste.
   * For each duplicate range name, keeps the first instance and re-creates
   * the rest with fresh uuid8 suffixes.
   */
  async fixDuplicateRanges(
    duplicateRanges: string[]
  ): Promise<number> {
    if (duplicateRanges.length === 0) return 0;

    const doc = await this.api.getDocument(this.docId);
    if (!doc.namedRanges) return 0;

    const requests: DocsRequest[] = [];
    let fixed = 0;

    for (const rangeName of duplicateRanges) {
      const entry = doc.namedRanges[rangeName];
      if (!entry) continue;

      const parsed = parseCitationRangeName(rangeName);
      if (!parsed) continue;

      const ranges = entry.namedRanges;
      if (ranges.length <= 1) continue;

      // Keep the first, re-id the rest
      for (let i = 1; i < ranges.length; i++) {
        const nr = ranges[i];
        const rangeInfo = nr.ranges?.[0];
        if (!rangeInfo) continue;

        // Delete old named range
        requests.push({
          deleteNamedRange: { namedRangeId: nr.namedRangeId },
        });
        // Create new one with fresh uuid8
        requests.push({
          createNamedRange: {
            name: buildCitationRangeName(parsed.paperId),
            range: {
              startIndex: rangeInfo.startIndex,
              endIndex: rangeInfo.endIndex,
            },
          },
        });
        fixed++;
      }
    }

    if (requests.length > 0) {
      await this.api.batchUpdate(this.docId, requests);
    }

    return fixed;
  }

  /**
   * Reformat all inCite citations in the document.
   * Finds all INCITE_CIT_* ranges, deletes old text, inserts new formatted text.
   * Processes in reverse index order to avoid shifting issues.
   */
  async reformatCitations(
    formatter: (paperId: string) => string | null
  ): Promise<{ updated: number; skipped: number }> {
    const doc = await this.api.getDocument(this.docId);
    if (!doc.namedRanges) return { updated: 0, skipped: 0 };

    // Collect all citation ranges with their paper IDs
    const citationRanges: { paperId: string; range: DocsNamedRange }[] = [];
    for (const [name, entry] of Object.entries(doc.namedRanges)) {
      const parsed = parseCitationRangeName(name);
      if (!parsed) continue;
      for (const namedRange of entry.namedRanges) {
        citationRanges.push({ paperId: parsed.paperId, range: namedRange });
      }
    }

    if (citationRanges.length === 0) return { updated: 0, skipped: 0 };

    // Sort by start index descending (process from end to avoid shifting)
    citationRanges.sort((a, b) => {
      const aStart = a.range.ranges[0]?.startIndex ?? 0;
      const bStart = b.range.ranges[0]?.startIndex ?? 0;
      return bStart - aStart;
    });

    const requests: DocsRequest[] = [];
    let updated = 0;
    let skipped = 0;

    for (const { paperId, range } of citationRanges) {
      const newText = formatter(paperId);
      if (!newText) {
        skipped++;
        continue;
      }

      const start = range.ranges[0].startIndex;
      const end = range.ranges[range.ranges.length - 1].endIndex;

      // Delete old named range
      requests.push({
        deleteNamedRange: { namedRangeId: range.namedRangeId },
      });
      // Delete old text
      requests.push({
        deleteContentRange: { range: { startIndex: start, endIndex: end } },
      });
      // Insert new text
      requests.push({
        insertText: { location: { index: start }, text: newText },
      });
      // Re-create named range with new uuid8 (auto-migrates legacy)
      requests.push({
        createNamedRange: {
          name: buildCitationRangeName(paperId),
          range: { startIndex: start, endIndex: start + newText.length },
        },
      });

      updated++;
    }

    if (requests.length > 0) {
      await this.api.batchUpdate(this.docId, requests);
    }

    return { updated, skipped };
  }

  /**
   * Remove all inCite named ranges and hyperlinks from citation text.
   * Keeps the text itself but strips the link styling and named ranges.
   */
  async cleanInciteData(): Promise<{ cleaned: number }> {
    const doc = await this.api.getDocument(this.docId);
    if (!doc.namedRanges) return { cleaned: 0 };

    const requests: DocsRequest[] = [];
    let cleaned = 0;

    for (const [name, entry] of Object.entries(doc.namedRanges)) {
      if (!parseCitationRangeName(name) && name !== BIBLIOGRAPHY_RANGE_NAME) {
        continue;
      }

      for (const namedRange of entry.namedRanges) {
        // Delete the named range
        requests.push({
          deleteNamedRange: { namedRangeId: namedRange.namedRangeId },
        });

        // Remove hyperlinks from the range (set link to empty)
        for (const r of namedRange.ranges) {
          requests.push({
            updateTextStyle: {
              range: { startIndex: r.startIndex, endIndex: r.endIndex },
              textStyle: { link: { url: "" } },
              fields: "link",
            },
          });
        }
        cleaned++;
      }
    }

    if (requests.length > 0) {
      await this.api.batchUpdate(this.docId, requests);
    }

    return { cleaned };
  }

  /** Find existing bibliography named ranges in the document. */
  private findBibliographyRange(doc: DocsDocument): DocsNamedRange[] {
    if (!doc.namedRanges?.[BIBLIOGRAPHY_RANGE_NAME]) return [];
    return doc.namedRanges[BIBLIOGRAPHY_RANGE_NAME].namedRanges;
  }
}
