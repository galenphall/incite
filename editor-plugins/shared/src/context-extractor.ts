/**
 * Extracts citation context around the cursor position.
 *
 * Grabs a window of text around the cursor, splits into sentences,
 * and returns the surrounding context with citation markers stripped.
 */

/** Split text into sentences using regex heuristics. */
export function splitSentences(text: string): string[] {
	// Split on sentence-ending punctuation followed by whitespace and uppercase.
	// Handles common abbreviations (Dr., Mr., etc., Fig., Eq., al.) by not
	// splitting after them.
	const abbrevs = /(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|etc|Fig|Figs|Eq|Eqs|al|vs|i\.e|e\.g|cf|no|vol|pp|ed|Rev|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.\s*/gi;

	// Replace abbreviation periods with a placeholder
	const placeholder = "\x00";
	let processed = text.replace(abbrevs, (match) =>
		match.replace(/\.\s*/, placeholder)
	);

	// Split on sentence boundaries
	const parts = processed.split(/(?<=[.!?])\s+(?=[A-Z"])/);

	// Restore abbreviation periods
	return parts
		.map((s) => s.replace(new RegExp(placeholder, "g"), ". ").trim())
		.filter((s) => s.length > 0);
}

/** Strip citation markers from text before sending to the API. */
export function stripCitations(text: string): string {
	return text
		.replace(/\[@[^\]]*\]/g, "")     // [@key]
		.replace(/\[cite\]/gi, "")        // [cite]
		.replace(/\\cite\{[^}]*\}/g, "")  // \cite{key}
		.replace(/\s{2,}/g, " ")          // collapse multiple spaces
		.trim();
}

export interface ExtractedContext {
	text: string;
	rawText: string; // before stripping citations
	cursorSentenceIndex: number;
	totalSentences: number;
}

/**
 * Extract context around the cursor position in the editor.
 *
 * @param fullText The full document text
 * @param cursorOffset Character offset of the cursor in the document
 * @param windowSentences Total number of sentences to include (split before/after cursor)
 * @returns Extracted context with citation markers stripped
 */
export function extractContext(
	fullText: string,
	cursorOffset: number,
	windowSentences: number
): ExtractedContext {
	// Grab a window of ~50 lines around cursor for performance
	const lines = fullText.split("\n");
	let charCount = 0;
	let cursorLine = 0;
	for (let i = 0; i < lines.length; i++) {
		charCount += lines[i].length + 1; // +1 for newline
		if (charCount > cursorOffset) {
			cursorLine = i;
			break;
		}
	}

	const windowLines = 50;
	const startLine = Math.max(0, cursorLine - windowLines);
	const endLine = Math.min(lines.length, cursorLine + windowLines);
	const windowText = lines.slice(startLine, endLine).join("\n");

	// Calculate cursor offset within the window
	let windowStartOffset = 0;
	for (let i = 0; i < startLine; i++) {
		windowStartOffset += lines[i].length + 1;
	}
	const windowCursorOffset = cursorOffset - windowStartOffset;

	// Split into sentences
	const sentences = splitSentences(windowText);
	if (sentences.length === 0) {
		return {
			text: stripCitations(windowText),
			rawText: windowText,
			cursorSentenceIndex: 0,
			totalSentences: 0,
		};
	}

	// Find which sentence the cursor is in
	let accum = 0;
	let cursorSentence = 0;
	for (let i = 0; i < sentences.length; i++) {
		const idx = windowText.indexOf(sentences[i], accum);
		if (idx === -1) continue;
		const sentEnd = idx + sentences[i].length;
		if (windowCursorOffset >= idx && windowCursorOffset <= sentEnd) {
			cursorSentence = i;
			break;
		}
		if (idx > windowCursorOffset) {
			// Cursor is before this sentence — use previous
			cursorSentence = Math.max(0, i - 1);
			break;
		}
		accum = sentEnd;
		cursorSentence = i; // fallback: last sentence
	}

	// Extract window: half before, half after (including cursor sentence)
	const halfWindow = Math.floor(windowSentences / 2);
	const start = Math.max(0, cursorSentence - halfWindow);
	const end = Math.min(sentences.length, cursorSentence + halfWindow + 1);

	const selectedSentences = sentences.slice(start, end);
	const rawText = selectedSentences.join(" ");

	return {
		text: stripCitations(rawText),
		rawText,
		cursorSentenceIndex: cursorSentence - start,
		totalSentences: selectedSentences.length,
	};
}
