import type { Recommendation } from "./types";

/** Detected citation style based on the template string. */
export type CitationStyle = "latex" | "pandoc" | "individual";

/**
 * Format a citation string from a recommendation using a template.
 *
 * Supports placeholders: {bibtex_key}, {paper_id}, {first_author},
 * {year}, {title}, {zotero_uri}, and ${...} variant for VS Code/LaTeX.
 */
export function formatCitation(
	rec: Recommendation,
	template: string
): string {
	const firstAuthor = rec.authors?.[0]
		? rec.authors[0].split(",")[0].split(" ").pop() ?? ""
		: "";

	const values: Record<string, string> = {
		bibtex_key: rec.bibtex_key ?? rec.paper_id,
		paper_id: rec.paper_id,
		first_author: firstAuthor,
		year: rec.year?.toString() ?? "",
		title: rec.title,
		zotero_uri: rec.zotero_uri ?? "",
	};

	let result = template;
	for (const [key, value] of Object.entries(values)) {
		// Support both {key} and ${key} placeholder styles
		result = result.replace(new RegExp(`\\{${key}\\}`, "g"), value);
		result = result.replace(new RegExp(`\\$\\{${key}\\}`, "g"), value);
	}
	return result;
}

/**
 * Detect citation style from a template string.
 *
 * - `\cite{...}` patterns → "latex"
 * - `[@...]` patterns → "pandoc"
 * - Everything else → "individual"
 */
export function detectCitationStyle(template: string): CitationStyle {
	if (/\\cite\{/.test(template)) return "latex";
	if (/\[@/.test(template)) return "pandoc";
	return "individual";
}

/**
 * Format multiple citations as a single grouped reference string.
 *
 * Auto-detects style from the template:
 * - LaTeX: `\cite{key1,key2,key3}`
 * - Pandoc: `[@key1; @key2; @key3]`
 * - Individual: each citation formatted separately, joined with separator
 */
export function formatMultiCitation(
	recs: Recommendation[],
	template: string,
	separator = "; "
): string {
	if (recs.length === 0) return "";
	if (recs.length === 1) return formatCitation(recs[0], template);

	const style = detectCitationStyle(template);

	switch (style) {
		case "latex": {
			const keys = recs.map((r) => r.bibtex_key ?? r.paper_id);
			return `\\cite{${keys.join(",")}}`;
		}
		case "pandoc": {
			const keys = recs.map((r) => r.bibtex_key ?? r.paper_id);
			return `[@${keys.join("; @")}]`;
		}
		case "individual": {
			return recs.map((r) => formatCitation(r, template)).join(separator);
		}
	}
}
