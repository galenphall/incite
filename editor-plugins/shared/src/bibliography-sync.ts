import type { TrackedCitation } from "./citation-tracker";
import { exportFormattedText } from "./bibliography";

/** Location of the bibliography section in a document. */
export interface BibliographySection {
	/** Line index of the heading (e.g. "## References"). */
	headingLine: number;
	/** Line index where content starts (first line after heading). */
	contentStart: number;
	/** Line index where content ends (exclusive). Next heading or EOF. */
	contentEnd: number;
}

const BIB_HEADING_RE = /^##\s+(references|bibliography)\s*$/i;
const HEADING_RE = /^(#{1,2})\s+/;

/**
 * Find the bibliography section in a markdown document.
 *
 * Looks for `## References` or `## Bibliography` (case-insensitive).
 * Returns the heading line and the content range up to the next
 * same-or-higher-level heading (## or #) or EOF.
 */
export function findBibliographySection(text: string): BibliographySection | null {
	const lines = text.split("\n");

	for (let i = 0; i < lines.length; i++) {
		if (!BIB_HEADING_RE.test(lines[i])) continue;

		const contentStart = i + 1;
		let contentEnd = lines.length;

		for (let j = contentStart; j < lines.length; j++) {
			const match = lines[j].match(HEADING_RE);
			if (match && match[1].length <= 2) {
				contentEnd = j;
				break;
			}
		}

		return { headingLine: i, contentStart, contentEnd };
	}

	return null;
}

/** Result of detecting an untracked citation in the document. */
export interface UntrackedCitation {
	/** Paper ID extracted from the fallback zotero URI. */
	paperId: string;
	/** The full markdown link text (e.g. "[Hall, 2024](zotero://...)"). */
	linkText: string;
}

/** Regex to match individual markdown links with zotero:// URIs. */
const ZOTERO_LINK_RE = /\[[^\]]+\]\(zotero:\/\/[^)]+\)/g;

/** Regex to extract paper_id from fallback zotero URI format. */
const FALLBACK_URI_RE = /zotero:\/\/select\/items\/0_([^)]+)/;

/**
 * Find tracked citations whose inserted text no longer appears in the document body.
 *
 * For citations with `insertedText`, searches for that exact string.
 * For older citations without it, searches for the paper_id in any zotero:// URI.
 *
 * @param body Document text ABOVE the bibliography section (excludes bibliography itself).
 * @param citations Currently tracked citations.
 * @returns Paper IDs of orphaned citations.
 */
export function scanForOrphans(body: string, citations: TrackedCitation[]): string[] {
	const orphans: string[] = [];

	for (const cite of citations) {
		if (cite.insertedText) {
			if (!body.includes(cite.insertedText)) {
				orphans.push(cite.paper_id);
			}
		} else {
			// Fallback: search for paper_id in any zotero:// URI in the body
			if (!body.includes(cite.paper_id)) {
				orphans.push(cite.paper_id);
			}
		}
	}

	return orphans;
}

/**
 * Find citation links in the document that are not in the tracker.
 *
 * Only detects citations using the fallback zotero URI format
 * (`zotero://select/items/0_{paper_id}`). Real Zotero URIs with
 * library item keys cannot be resolved to paper IDs without an API call.
 *
 * @param body Document text ABOVE the bibliography section.
 * @param citations Currently tracked citations.
 * @returns Untracked citations with extracted paper IDs.
 */
export function scanForUntracked(
	body: string,
	citations: TrackedCitation[]
): UntrackedCitation[] {
	const trackedIds = new Set(citations.map((c) => c.paper_id));
	const untracked: UntrackedCitation[] = [];
	const seen = new Set<string>();

	for (const match of body.matchAll(ZOTERO_LINK_RE)) {
		const linkText = match[0];
		const uriMatch = linkText.match(FALLBACK_URI_RE);
		if (!uriMatch) continue; // real Zotero URI — skip

		const paperId = uriMatch[1];
		if (trackedIds.has(paperId) || seen.has(paperId)) continue;

		seen.add(paperId);
		untracked.push({ paperId, linkText });
	}

	return untracked;
}

/**
 * Format bibliography content from tracked citations.
 *
 * Returns APA-formatted text with a leading newline (for spacing after the heading).
 * Returns just "\n" if there are no citations.
 */
export function formatBibliographyContent(citations: TrackedCitation[]): string {
	if (citations.length === 0) return "\n";
	return "\n" + exportFormattedText(citations) + "\n";
}

/**
 * Get the document body text above the bibliography section.
 *
 * If no bibliography section exists, returns the full document text.
 * Used to scope orphan/untracked scanning to the main content area.
 */
export function getBodyText(text: string): string {
	const section = findBibliographySection(text);
	if (!section) return text;

	const lines = text.split("\n");
	return lines.slice(0, section.headingLine).join("\n");
}
