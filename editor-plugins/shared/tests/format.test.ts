import { describe, it, expect } from "vitest";
import {
	formatCitation,
	formatMultiCitation,
	detectCitationStyle,
} from "../src/format";
import type { Recommendation } from "../src/types";

/** Helper to create a minimal Recommendation for testing. */
function makeRec(overrides: Partial<Recommendation>): Recommendation {
	return {
		paper_id: "p1",
		rank: 1,
		score: 0.9,
		title: "Test Paper",
		authors: ["Alice Smith"],
		year: 2024,
		bibtex_key: "Smith2024",
		zotero_uri: "zotero://select/items/ABC123",
		...overrides,
	};
}

const rec1 = makeRec({
	paper_id: "p1",
	authors: ["Alice Smith"],
	year: 2024,
	bibtex_key: "Smith2024",
	zotero_uri: "zotero://select/items/ABC123",
	title: "Paper One",
});

const rec2 = makeRec({
	paper_id: "p2",
	authors: ["Bob Jones"],
	year: 2023,
	bibtex_key: "Jones2023",
	zotero_uri: "zotero://select/items/DEF456",
	title: "Paper Two",
	rank: 2,
	score: 0.8,
});

const rec3 = makeRec({
	paper_id: "p3",
	authors: ["Carol Lee"],
	year: 2022,
	bibtex_key: "Lee2022",
	zotero_uri: "zotero://select/items/GHI789",
	title: "Paper Three",
	rank: 3,
	score: 0.7,
});

// ---------- Markdown multi-citation tests ----------

describe("formatMultiCitation — markdown link templates", () => {
	it("produces per-paper hyperlinks with outer parens grouping", () => {
		const template = "[({first_author}, {year})]({zotero_uri})";
		const result = formatMultiCitation([rec1, rec2], template);
		expect(result).toBe(
			"([Smith, 2024](zotero://select/items/ABC123); [Jones, 2023](zotero://select/items/DEF456))"
		);
	});

	it("produces per-paper hyperlinks with outer brackets grouping", () => {
		const template = "[[{first_author}, {year}]]({zotero_uri})";
		const result = formatMultiCitation([rec1, rec2], template);
		expect(result).toBe(
			"[[Smith, 2024](zotero://select/items/ABC123); [Jones, 2023](zotero://select/items/DEF456)]"
		);
	});

	it("joins without grouping when text has no outer delimiters", () => {
		const template = "[{first_author} {year}]({zotero_uri})";
		const result = formatMultiCitation([rec1, rec2], template);
		expect(result).toBe(
			"[Smith 2024](zotero://select/items/ABC123); [Jones 2023](zotero://select/items/DEF456)"
		);
	});

	it("works with three papers", () => {
		const template = "[({first_author}, {year})]({zotero_uri})";
		const result = formatMultiCitation([rec1, rec2, rec3], template);
		expect(result).toBe(
			"([Smith, 2024](zotero://select/items/ABC123); " +
				"[Jones, 2023](zotero://select/items/DEF456); " +
				"[Lee, 2022](zotero://select/items/GHI789))"
		);
	});

	it("respects custom separator", () => {
		const template = "[({first_author}, {year})]({zotero_uri})";
		const result = formatMultiCitation([rec1, rec2], template, ", ");
		expect(result).toBe(
			"([Smith, 2024](zotero://select/items/ABC123), [Jones, 2023](zotero://select/items/DEF456))"
		);
	});

	it("returns single formatted citation unchanged for one paper", () => {
		const template = "[({first_author}, {year})]({zotero_uri})";
		const result = formatMultiCitation([rec1], template);
		// Single citation goes through formatCitation directly (line 70)
		expect(result).toBe("[(Smith, 2024)](zotero://select/items/ABC123)");
	});
});

// ---------- Non-markdown templates still work ----------

describe("formatMultiCitation — non-markdown templates (regression)", () => {
	it("groups individual citations with parens", () => {
		const template = "({first_author}, {year})";
		const result = formatMultiCitation([rec1, rec2], template);
		expect(result).toBe("(Smith, 2024; Jones, 2023)");
	});

	it("groups individual citations with brackets", () => {
		const template = "[{first_author}, {year}]";
		const result = formatMultiCitation([rec1, rec2], template);
		expect(result).toBe("[Smith, 2024; Jones, 2023]");
	});

	it("joins individual citations without delimiters", () => {
		const template = "{first_author} {year}";
		const result = formatMultiCitation([rec1, rec2], template);
		expect(result).toBe("Smith 2024; Jones 2023");
	});

	it("formats LaTeX multi-citation", () => {
		const template = "\\cite{{bibtex_key}}";
		const result = formatMultiCitation([rec1, rec2], template);
		expect(result).toBe("\\cite{Smith2024,Jones2023}");
	});

	it("formats Pandoc multi-citation", () => {
		const template = "[@{bibtex_key}]";
		const result = formatMultiCitation([rec1, rec2], template);
		expect(result).toBe("[@Smith2024; @Jones2023]");
	});

	it("returns empty string for no recommendations", () => {
		expect(formatMultiCitation([], "({first_author}, {year})")).toBe("");
	});
});

// ---------- detectCitationStyle ----------

describe("detectCitationStyle", () => {
	it("detects LaTeX", () => {
		expect(detectCitationStyle("\\cite{{bibtex_key}}")).toBe("latex");
	});

	it("detects Pandoc", () => {
		expect(detectCitationStyle("[@{bibtex_key}]")).toBe("pandoc");
	});

	it("detects individual for plain templates", () => {
		expect(detectCitationStyle("({first_author}, {year})")).toBe(
			"individual"
		);
	});

	it("detects individual for markdown link templates", () => {
		expect(
			detectCitationStyle("[({first_author}, {year})]({zotero_uri})")
		).toBe("individual");
	});
});

// ---------- formatCitation (single) ----------

describe("formatCitation", () => {
	it("formats a markdown link template for a single paper", () => {
		const template = "[({first_author}, {year})]({zotero_uri})";
		const result = formatCitation(rec1, template);
		expect(result).toBe("[(Smith, 2024)](zotero://select/items/ABC123)");
	});

	it("substitutes all placeholders", () => {
		const template = "{title} by {first_author} ({year})";
		const result = formatCitation(rec1, template);
		expect(result).toBe("Paper One by Smith (2024)");
	});
});
