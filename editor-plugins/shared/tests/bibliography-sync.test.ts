import { describe, it, expect } from "vitest";
import type { TrackedCitation } from "../src/citation-tracker";
import {
	findBibliographySection,
	scanForOrphans,
	scanForUntracked,
	formatBibliographyContent,
	getBodyText,
} from "../src/bibliography-sync";

function makeCitation(overrides: Partial<TrackedCitation> & { paper_id: string }): TrackedCitation {
	return {
		bibtex_key: overrides.paper_id,
		title: "Test Paper",
		authors: ["Test Author"],
		year: 2024,
		insertedAt: Date.now(),
		...overrides,
	};
}

describe("findBibliographySection", () => {
	it("finds ## References heading with content until EOF", () => {
		const text = [
			"# My Paper",
			"",
			"Some text.",
			"",
			"## References",
			"",
			"Hall (2024). Title. *Journal*.",
		].join("\n");

		const result = findBibliographySection(text);
		expect(result).not.toBeNull();
		expect(result!.headingLine).toBe(4);
		expect(result!.contentStart).toBe(5);
		expect(result!.contentEnd).toBe(7); // exclusive, = total line count
	});

	it("finds ## Bibliography heading", () => {
		const text = "# Paper\n\n## Bibliography\n\nEntry here.\n";
		const result = findBibliographySection(text);
		expect(result).not.toBeNull();
		expect(result!.headingLine).toBe(2);
	});

	it("stops at next same-level heading", () => {
		const text = [
			"## References",
			"",
			"Entry 1.",
			"",
			"## Appendix",
			"",
			"Extra stuff.",
		].join("\n");

		const result = findBibliographySection(text);
		expect(result).not.toBeNull();
		expect(result!.contentEnd).toBe(4); // stops before ## Appendix
	});

	it("stops at higher-level heading", () => {
		const text = [
			"## References",
			"",
			"Entry 1.",
			"",
			"# Next Chapter",
		].join("\n");

		const result = findBibliographySection(text);
		expect(result).not.toBeNull();
		expect(result!.contentEnd).toBe(4);
	});

	it("does not stop at lower-level heading (###)", () => {
		const text = [
			"## References",
			"",
			"### Primary Sources",
			"",
			"Entry 1.",
		].join("\n");

		const result = findBibliographySection(text);
		expect(result).not.toBeNull();
		expect(result!.contentEnd).toBe(5);
	});

	it("returns null when no bibliography heading exists", () => {
		const text = "# My Paper\n\nJust some text.\n";
		expect(findBibliographySection(text)).toBeNull();
	});

	it("is case-insensitive", () => {
		const text = "## references\n\nEntry.\n";
		const result = findBibliographySection(text);
		expect(result).not.toBeNull();
	});
});

describe("scanForOrphans", () => {
	it("returns empty when all citations found in body", () => {
		const body = "Some text [Hall, 2024](zotero://select/items/0_abc123) more text.";
		const citations = [
			makeCitation({ paper_id: "abc123", insertedText: "[Hall, 2024](zotero://select/items/0_abc123)" }),
		];
		expect(scanForOrphans(body, citations)).toEqual([]);
	});

	it("detects orphaned citation by insertedText", () => {
		const body = "Some text without any citations.";
		const citations = [
			makeCitation({ paper_id: "abc123", insertedText: "[Hall, 2024](zotero://select/items/0_abc123)" }),
		];
		expect(scanForOrphans(body, citations)).toEqual(["abc123"]);
	});

	it("falls back to paper_id search in zotero URIs when no insertedText", () => {
		const body = "Text with [Hall, 2024](zotero://select/items/0_abc123) here.";
		const citations = [
			makeCitation({ paper_id: "abc123" }), // no insertedText
		];
		expect(scanForOrphans(body, citations)).toEqual([]);
	});

	it("detects orphan via fallback when paper_id not in any URI", () => {
		const body = "Text with no matching citations.";
		const citations = [
			makeCitation({ paper_id: "abc123" }), // no insertedText
		];
		expect(scanForOrphans(body, citations)).toEqual(["abc123"]);
	});
});

describe("scanForUntracked", () => {
	it("returns empty when all links are tracked", () => {
		const body = "Text [Hall, 2024](zotero://select/items/0_abc123) end.";
		const citations = [makeCitation({ paper_id: "abc123" })];
		expect(scanForUntracked(body, citations)).toEqual([]);
	});

	it("detects untracked citation from fallback zotero URI", () => {
		const body = "Pasted [Smith, 2020](zotero://select/items/0_xyz789) here.";
		const citations: TrackedCitation[] = [];
		const result = scanForUntracked(body, citations);
		expect(result).toHaveLength(1);
		expect(result[0].paperId).toBe("xyz789");
		expect(result[0].linkText).toBe("[Smith, 2020](zotero://select/items/0_xyz789)");
	});

	it("detects multiple untracked citations in multi-citation group", () => {
		const body = "([Jones, 2023](zotero://select/items/0_aaa); [Smith, 2020](zotero://select/items/0_bbb))";
		const citations: TrackedCitation[] = [];
		const result = scanForUntracked(body, citations);
		expect(result).toHaveLength(2);
		expect(result.map((r) => r.paperId)).toEqual(["aaa", "bbb"]);
	});

	it("ignores links that are already tracked", () => {
		const body = "([Jones, 2023](zotero://select/items/0_aaa); [Smith, 2020](zotero://select/items/0_bbb))";
		const citations = [makeCitation({ paper_id: "aaa" })];
		const result = scanForUntracked(body, citations);
		expect(result).toHaveLength(1);
		expect(result[0].paperId).toBe("bbb");
	});

	it("ignores non-fallback zotero URIs (real Zotero keys)", () => {
		const body = "[Hall, 2024](zotero://select/library/items/ABCD1234)";
		const citations: TrackedCitation[] = [];
		expect(scanForUntracked(body, citations)).toEqual([]);
	});
});

describe("formatBibliographyContent", () => {
	it("produces APA-formatted entries with blank line prefix", () => {
		const citations = [
			makeCitation({ paper_id: "a", title: "Alpha Paper", authors: ["Smith, John"], year: 2024 }),
		];
		const result = formatBibliographyContent(citations);
		expect(result).toContain("Smith (2024). Alpha Paper.");
		expect(result.startsWith("\n")).toBe(true); // blank line after heading
	});

	it("returns single newline for empty citations", () => {
		expect(formatBibliographyContent([])).toBe("\n");
	});
});

describe("getBodyText", () => {
	it("returns text above bibliography heading", () => {
		const text = "Body text here.\n\n## References\n\nEntry.";
		const body = getBodyText(text);
		expect(body).toBe("Body text here.\n");
	});

	it("returns full text when no bibliography section exists", () => {
		const text = "Just body text.";
		expect(getBodyText(text)).toBe("Just body text.");
	});
});
