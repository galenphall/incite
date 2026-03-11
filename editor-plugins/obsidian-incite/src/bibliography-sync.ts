import { Editor, Notice } from "obsidian";
import type { TrackedCitation } from "@incite/shared";
import {
	findBibliographySection,
	scanForOrphans,
	scanForUntracked,
	formatBibliographyContent,
	getBodyText,
} from "@incite/shared";
import type { CitationTracker } from "@incite/shared";

const BIB_HEADING = "## References";

/**
 * Rewrite the bibliography section in the editor from current tracker state.
 *
 * If no bibliography section exists, appends one at the end of the document.
 */
export function updateBibliography(editor: Editor, tracker: CitationTracker): void {
	const text = editor.getValue();
	const citations = tracker.getAll();
	const newContent = formatBibliographyContent(citations);
	const section = findBibliographySection(text);

	if (section) {
		// Replace existing bibliography content (keep heading)
		const lines = text.split("\n");
		const before = lines.slice(0, section.contentStart).join("\n");
		const after = lines.slice(section.contentEnd).join("\n");
		const newText = before + "\n" + newContent.trimStart() + (after ? "\n" + after : "");
		// Only update if changed to avoid unnecessary edits
		if (newText !== text) {
			const cursor = editor.getCursor();
			editor.setValue(newText);
			editor.setCursor(cursor);
		}
	} else if (citations.length > 0) {
		// Append bibliography section at end of document
		const suffix = text.endsWith("\n") ? "\n" : "\n\n";
		const newText = text + suffix + BIB_HEADING + "\n" + newContent;
		const cursor = editor.getCursor();
		editor.setValue(newText);
		editor.setCursor(cursor);
	}
}

/**
 * Run full reconciliation: detect orphans and untracked citations,
 * update tracker and bibliography, show notices.
 *
 * @returns true if any changes were made.
 */
export async function reconcile(
	editor: Editor,
	tracker: CitationTracker
): Promise<boolean> {
	const text = editor.getValue();
	const body = getBodyText(text);
	const citations = tracker.getAll();
	let changed = false;

	// 1. Detect and remove orphans
	const orphanIds = scanForOrphans(body, citations);
	for (const paperId of orphanIds) {
		const cite = citations.find((c) => c.paper_id === paperId);
		const label = cite
			? `${cite.authors[0]?.split(/\s+/).pop() ?? ""} ${cite.year ?? ""}`.trim()
			: paperId;
		await tracker.remove(paperId);
		new Notice(`Removed "${label}" from bibliography \u2014 no longer cited.`);
		changed = true;
	}

	// 2. Detect untracked citations (pasted from other documents)
	const untracked = scanForUntracked(body, tracker.getAll());
	if (untracked.length > 0) {
		for (const { paperId, linkText } of untracked) {
			// Create a minimal TrackedCitation from the link text
			const authorYearMatch = linkText.match(/\[([^\],]+?)(?:,\s*(\d{4}))?\]/);
			const author = authorYearMatch?.[1] ?? "Unknown";
			const year = authorYearMatch?.[2] ? parseInt(authorYearMatch[2]) : undefined;

			const minimalCitation: TrackedCitation = {
				paper_id: paperId,
				bibtex_key: paperId,
				title: `${author}${year ? ` (${year})` : ""}`,
				authors: [author],
				year,
				insertedAt: Date.now(),
				insertedText: linkText,
			};

			// Track using trackCitation — we build the citation directly
			// since we don't have a full Recommendation object
			await tracker.trackCitation(minimalCitation);
		}
		const count = untracked.length;
		new Notice(`Added ${count} citation${count > 1 ? "s" : ""} to bibliography.`);
		changed = true;
	}

	// 3. Update bibliography if anything changed
	if (changed) {
		updateBibliography(editor, tracker);
	}

	return changed;
}
