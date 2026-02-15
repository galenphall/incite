import type { TrackedCitation } from "./citation-tracker";

// --- LaTeX escaping ---

const LATEX_SPECIAL: Record<string, string> = {
	"\\": "\\textbackslash{}",
	"{": "\\{",
	"}": "\\}",
	"&": "\\&",
	"%": "\\%",
	"#": "\\#",
	_: "\\_",
	"~": "\\textasciitilde{}",
	"^": "\\textasciicircum{}",
};

/** Escape special LaTeX characters in a string. */
export function escapeLaTeX(text: string): string {
	let result = "";
	for (const ch of text) {
		result += LATEX_SPECIAL[ch] ?? ch;
	}
	return result;
}

// --- BibTeX export ---

function formatBibTeXEntry(cite: TrackedCitation): string {
	const key = cite.bibtex_key || cite.paper_id;
	const fields: string[] = [];

	fields.push(`  title = {${escapeLaTeX(cite.title)}}`);

	if (cite.authors.length > 0) {
		fields.push(`  author = {${escapeLaTeX(cite.authors.join(" and "))}}`);
	}

	if (cite.year != null) {
		fields.push(`  year = {${cite.year}}`);
	}

	if (cite.journal) {
		fields.push(`  journal = {${escapeLaTeX(cite.journal)}}`);
	}

	if (cite.doi) {
		fields.push(`  doi = {${cite.doi}}`);
	}

	if (cite.abstract) {
		fields.push(`  abstract = {${escapeLaTeX(cite.abstract)}}`);
	}

	return `@article{${key},\n${fields.join(",\n")}\n}`;
}

/** Export tracked citations as a BibTeX string. */
export function exportBibTeX(citations: TrackedCitation[]): string {
	return citations.map(formatBibTeXEntry).join("\n\n");
}

// --- RIS export ---

function formatRISEntry(cite: TrackedCitation): string {
	const lines: string[] = [];
	lines.push("TY  - JOUR");
	lines.push(`TI  - ${cite.title}`);

	for (const author of cite.authors) {
		lines.push(`AU  - ${author}`);
	}

	if (cite.year != null) {
		lines.push(`PY  - ${cite.year}`);
	}

	if (cite.doi) {
		lines.push(`DO  - ${cite.doi}`);
	}

	if (cite.journal) {
		lines.push(`JO  - ${cite.journal}`);
	}

	if (cite.abstract) {
		lines.push(`AB  - ${cite.abstract}`);
	}

	lines.push("ER  - ");
	return lines.join("\n");
}

/** Export tracked citations as a RIS string. */
export function exportRIS(citations: TrackedCitation[]): string {
	return citations.map(formatRISEntry).join("\n\n");
}

// --- Formatted text (APA-style) ---

function getLastName(author: string): string {
	if (author.includes(",")) {
		return author.split(",")[0].trim();
	}
	const parts = author.trim().split(/\s+/);
	return parts[parts.length - 1];
}

function formatAPAAuthors(authors: string[]): string {
	if (authors.length === 0) return "";
	if (authors.length === 1) return getLastName(authors[0]);
	if (authors.length === 2) {
		return `${getLastName(authors[0])} & ${getLastName(authors[1])}`;
	}
	// 3+ authors: First et al. (APA 7th)
	return `${getLastName(authors[0])} et al.`;
}

function formatAPAEntry(cite: TrackedCitation): string {
	const parts: string[] = [];

	// Authors
	if (cite.authors.length > 0) {
		if (cite.authors.length <= 2) {
			parts.push(cite.authors.map(getLastName).join(" & "));
		} else {
			const lastNames = cite.authors.map(getLastName);
			const allButLast = lastNames.slice(0, -1).join(", ");
			parts.push(`${allButLast}, & ${lastNames[lastNames.length - 1]}`);
		}
	}

	// Year
	parts.push(cite.year != null ? `(${cite.year}).` : "(n.d.).");

	// Title
	parts.push(`${cite.title}.`);

	// Journal
	if (cite.journal) {
		parts.push(`${cite.journal}.`);
	}

	// DOI
	if (cite.doi) {
		parts.push(`https://doi.org/${cite.doi}`);
	}

	return parts.join(" ");
}

/** Export tracked citations as APA-style formatted text, sorted by first author. */
export function exportFormattedText(citations: TrackedCitation[]): string {
	const sorted = [...citations].sort((a, b) => {
		const aName = a.authors.length > 0 ? getLastName(a.authors[0]) : "";
		const bName = b.authors.length > 0 ? getLastName(b.authors[0]) : "";
		return aName.localeCompare(bName);
	});
	return sorted.map(formatAPAEntry).join("\n\n");
}
