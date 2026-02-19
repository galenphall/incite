/**
 * Word document context extraction.
 *
 * Reads the current selection and surrounding paragraphs from the Word
 * document via Office.js, then delegates to the shared context extractor.
 */

import { extractContext } from "@incite/shared";
import { settings } from "./settings";

/* global Word */

declare const Word: {
	run: <T>(callback: (context: WordContext) => Promise<T>) => Promise<T>;
};

interface WordContext {
	document: {
		getSelection: () => WordRange;
		body: { paragraphs: WordParagraphCollection };
	};
	sync: () => Promise<void>;
}

interface WordRange {
	text: string;
	paragraphs: WordParagraphCollection;
	load: (props: string) => void;
	insertText: (text: string, location: string) => void;
}

interface WordParagraphCollection {
	items: Array<{ text: string }>;
	load: (props: string) => void;
}

/**
 * Extract context text from the Word document around the cursor.
 *
 * Reads all document paragraphs, locates the cursor position by matching
 * the selection paragraph, then delegates to the shared `extractContext()`.
 */
export async function getContextFromWord(): Promise<{
	text: string;
	cursorSentenceIndex: number;
}> {
	return Word.run(async (context) => {
		const selection = context.document.getSelection();
		selection.load("text");

		// Load the paragraph(s) containing the selection/cursor
		const selectionParagraphs = selection.paragraphs;
		selectionParagraphs.load("text");

		const paragraphs = context.document.body.paragraphs;
		paragraphs.load("text");

		await context.sync();

		const allText = paragraphs.items.map((p) => p.text).join("\n");
		const selectionText = selection.text;

		// Find cursor position using the selection's paragraph location
		let cursorOffset: number;

		// Identify which document paragraph contains the cursor/selection.
		// Match the first selection paragraph against the document paragraphs.
		const cursorParaText =
			selectionParagraphs.items.length > 0
				? selectionParagraphs.items[0].text
				: "";

		// Find the paragraph index -- if multiple paragraphs share the same text,
		// pick the first match. This is rare in practice and still far better
		// than defaulting to the middle of the document.
		let cursorParaIndex = -1;
		let charPos = 0;
		for (let i = 0; i < paragraphs.items.length; i++) {
			if (paragraphs.items[i].text === cursorParaText) {
				cursorParaIndex = i;
				break;
			}
			charPos += paragraphs.items[i].text.length + 1; // +1 for \n
		}

		if (cursorParaIndex >= 0) {
			if (selectionText.trim().length > 0) {
				// Text is selected -- find it within the matched paragraph
				const paraStart = charPos;
				const localIdx = cursorParaText.indexOf(selectionText);
				cursorOffset =
					localIdx >= 0 ? paraStart + localIdx : paraStart;
			} else {
				// No selection (cursor only) -- use the start of the matched paragraph
				cursorOffset = charPos;
			}
		} else {
			// Fallback: couldn't match paragraph. Use selection text search
			// or middle of document.
			if (selectionText.trim().length > 0) {
				const idx = allText.indexOf(selectionText);
				cursorOffset = idx >= 0 ? idx : Math.floor(allText.length / 2);
			} else {
				cursorOffset = Math.floor(allText.length / 2);
			}
		}

		const extracted = extractContext(
			allText,
			cursorOffset,
			settings.contextSentences,
		);
		return {
			text: extracted.text,
			cursorSentenceIndex: extracted.cursorSentenceIndex,
		};
	});
}
