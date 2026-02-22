/**
 * Reads the user's Zotero library via the privileged JS API.
 * Returns paper metadata and PDF attachment paths.
 */

export interface ZoteroPaper {
	id: string;
	title: string;
	abstract: string;
	authors: string[];
	year: number | null;
	doi: string | null;
	journal: string | null;
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
		if (att.attachmentContentType === "application/pdf") {
			const filePath = await att.getFilePathAsync();
			if (filePath) return filePath;
		}
	}
	return null;
}

/**
 * Read all papers from the user's Zotero library.
 * Filters to regular paper types and extracts metadata + PDF paths.
 */
export async function readZoteroLibrary(): Promise<ZoteroPaper[]> {
	const libraryID = Zotero.Libraries.userLibraryID;
	const allItems = await Zotero.Items.getAll(libraryID, true, false);

	const papers: ZoteroPaper[] = [];

	for (const item of allItems) {
		// Skip non-paper types
		if (item.isNote() || item.isAnnotation() || item.isAttachment()) continue;

		const typeName = Zotero.ItemTypes.getName(item.itemTypeID);
		if (!PAPER_TYPES.has(typeName)) continue;

		const title = item.getField("title").trim();
		if (!title) continue;

		// Extract authors (last names of creator type "author")
		const creators = item.getCreators();
		const authors: string[] = [];
		for (const c of creators) {
			if (c.creatorType === "author" && c.lastName) {
				authors.push(c.lastName);
			}
		}

		const pdfPath = await findPdfPath(item);

		papers.push({
			id: item.key,
			title,
			abstract: item.getField("abstractNote") || "",
			authors,
			year: parseYear(item.getField("date") || ""),
			doi: item.getField("DOI") || null,
			journal: item.getField("publicationTitle") || null,
			pdfPath,
		});
	}

	return papers;
}
