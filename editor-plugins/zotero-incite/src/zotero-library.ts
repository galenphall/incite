/**
 * Reads the user's Zotero library via the privileged JS API.
 * Returns paper metadata and PDF attachment paths.
 */

export interface ZoteroPaper {
	id: string;
	title: string;
	abstract: string;
	authors: string[];
	structured_authors: { given?: string; family: string }[];
	year: number | null;
	doi: string | null;
	journal: string | null;
	volume: string | null;
	issue: string | null;
	pages: string | null;
	item_type: string;
	pdfPath: string | null;
}

/** Item types we consider "papers" (matches the Python zotero_reader). */
const PAPER_TYPES = new Set([
	"journalArticle",
	"conferencePaper",
	"preprint",
	"thesis",
	"book",
	"bookSection",
	"report",
	"manuscript",
]);

/** Maps Zotero itemType to CSL type strings. */
const ZOTERO_TYPE_TO_CSL: Record<string, string> = {
	journalArticle: "article-journal",
	book: "book",
	bookSection: "chapter",
	conferencePaper: "paper-conference",
	thesis: "thesis",
	preprint: "article",
	report: "report",
	manuscript: "manuscript",
};

/** Safely get a field value, returning "" if the field doesn't exist for this item type. */
function safeGetField(item: Zotero.Item, field: string): string {
	try {
		return (item.getField(field) as string) || "";
	} catch {
		return "";
	}
}

/** Parse a 4-digit year from a Zotero date string, or return null. */
function parseYear(dateStr: string): number | null {
	const match = dateStr.match(/^\d{4}/);
	if (!match) return null;
	const year = parseInt(match[0], 10);
	return isNaN(year) ? null : year;
}

/** Find the first PDF attachment path for an item, or null. */
async function findPdfPath(item: Zotero.Item): Promise<string | null> {
	const attachmentIDs = item.getAttachments();
	for (const attId of attachmentIDs) {
		const att = await Zotero.Items.getAsync(attId);
		// Check MIME type first, then fall back to file extension —
		// some attachments have missing or generic MIME types
		const isPdf =
			att.attachmentContentType === "application/pdf" ||
			(att.attachmentFilename?.toLowerCase().endsWith(".pdf") ?? false);
		if (isPdf) {
			const filePath = await att.getFilePathAsync();
			if (filePath) return filePath;
		}
	}
	return null;
}

/**
 * Read all papers from the user's Zotero libraries (personal + group).
 * Filters to regular paper types and extracts metadata + PDF paths.
 * Deduplicates by DOI so the same paper isn't uploaded twice.
 */
export async function readZoteroLibrary(): Promise<ZoteroPaper[]> {
	// Collect items from personal library, plus group libraries if enabled
	const includeGroups = Zotero.Prefs.get("extensions.incite.includeGroupLibraries", true);
	const libraries = includeGroups
		? Zotero.Libraries.getAll()
		: [Zotero.Libraries.get(Zotero.Libraries.userLibraryID)];
	let allItems: Zotero.Item[] = [];
	for (const lib of libraries) {
		const items = await Zotero.Items.getAll(lib.libraryID, true, false);
		allItems = allItems.concat(items);
	}

	const papers: ZoteroPaper[] = [];
	const seenDois = new Set<string>();

	for (const item of allItems) {
		// Skip non-paper types
		if (item.isNote() || item.isAnnotation() || item.isAttachment()) continue;

		const typeName = Zotero.ItemTypes.getName(item.itemTypeID);
		if (!PAPER_TYPES.has(typeName)) continue;

		const title = item.getField("title").trim();
		if (!title) continue;

		// Extract full author names (first + last) for authors and editors
		// Note: getCreators() returns creatorTypeID (integer), not creatorType (string)
		const creators = item.getCreators();
		const authors: string[] = [];
		const structured_authors: { given?: string; family: string }[] = [];
		for (const c of creators) {
			const cTypeName = Zotero.CreatorTypes.getName(c.creatorTypeID);
			if ((cTypeName === "author" || cTypeName === "editor") && c.lastName) {
				const name = c.firstName ? `${c.firstName} ${c.lastName}` : c.lastName;
				authors.push(name);
				structured_authors.push(
					c.firstName
						? { given: c.firstName, family: c.lastName }
						: { family: c.lastName },
				);
			}
		}

		const doi = item.getField("DOI") || null;

		// Deduplicate by DOI across libraries
		if (doi) {
			const normDoi = doi.toLowerCase().trim();
			if (seenDois.has(normDoi)) continue;
			seenDois.add(normDoi);
		}

		const pdfPath = await findPdfPath(item);

		papers.push({
			id: item.key,
			title,
			abstract: safeGetField(item, "abstractNote"),
			authors,
			structured_authors,
			year: parseYear(safeGetField(item, "date")),
			doi,
			journal: safeGetField(item, "publicationTitle") || null,
			volume: safeGetField(item, "volume") || null,
			issue: safeGetField(item, "issue") || null,
			pages: safeGetField(item, "pages") || null,
			item_type: ZOTERO_TYPE_TO_CSL[typeName] ?? "article-journal",
			pdfPath,
		});
	}

	return papers;
}
