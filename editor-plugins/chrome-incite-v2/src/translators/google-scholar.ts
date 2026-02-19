import type { Translator, PaperMetadata, DetectionResult } from "./types";

export const googleScholarTranslator: Translator = {
  name: "google-scholar",
  urlPatterns: [/scholar\.google\.\w+\/scholar/],

  detect(doc: Document): DetectionResult | null {
    const results = doc.querySelectorAll(".gs_ri");
    return results.length > 0 ? { type: "multiple" } : null;
  },

  extractSingle(_doc: Document): PaperMetadata | null {
    return null; // Scholar is always a search results page
  },

  extractMultiple(doc: Document): PaperMetadata[] {
    const results: PaperMetadata[] = [];
    const items = doc.querySelectorAll(".gs_ri");

    for (const item of items) {
      try {
        // Title from h3 > a
        const titleEl = item.querySelector("h3 a");
        const title = titleEl?.textContent?.trim();
        if (!title) continue;

        // URL from title link
        const url = titleEl?.getAttribute("href") ?? undefined;

        // Authors, year, journal from .gs_a line
        // Format: "Author1, Author2 - Journal, Year - Publisher"
        const metaLine = item.querySelector(".gs_a")?.textContent ?? "";
        const parts = metaLine.split(" - ");

        let authors: string[] | undefined;
        let year: number | undefined;
        let journal: string | undefined;

        if (parts.length >= 1) {
          const authorStr = parts[0].trim();
          // Remove trailing ellipsis and split by comma
          authors = authorStr.replace(/…$/, "").split(",")
            .map(a => a.trim())
            .filter(a => a.length > 0 && !a.match(/^\d{4}$/));
          if (!authors.length) authors = undefined;
        }

        if (parts.length >= 2) {
          const yearMatch = parts[1].match(/(\d{4})/);
          year = yearMatch ? parseInt(yearMatch[1], 10) : undefined;
          // Journal is text before the year
          const journalPart = parts[1].replace(/,?\s*\d{4}.*$/, "").trim();
          journal = journalPart || undefined;
        }

        // Snippet from .gs_rs
        const snippet = item.querySelector(".gs_rs")?.textContent?.trim() ?? undefined;

        // PDF link from sibling .gs_ggs element (contains [PDF] links)
        const parentResult = item.closest(".gs_r");
        const pdfLink = parentResult?.querySelector<HTMLAnchorElement>(
          ".gs_ggs a, .gs_or_ggsm a"
        );
        const pdf_url = pdfLink?.href ?? undefined;

        results.push({
          title,
          authors,
          year,
          journal,
          url,
          abstract: snippet, // Scholar snippet is partial abstract at best
          pdf_url,
        });
      } catch { /* skip malformed result */ }
    }

    return results;
  },
};
