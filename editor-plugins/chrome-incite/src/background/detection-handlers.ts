/**
 * Paper detection handler.
 *
 * Detects academic papers on the active tab by first checking the cache,
 * then trying the content script, and finally falling back to inline
 * extraction via chrome.scripting.executeScript.
 */
import type { PaperMetadata } from "../translators/types";
import { getActiveTab } from "./client";
import { detectedPapers } from "./state";
import { updateBadge } from "./library-handlers";

/**
 * IMPORTANT: The function below runs in the PAGE context via
 * chrome.scripting.executeScript. It cannot import modules — everything
 * (getMeta, getAllMeta, extractStructuredText, etc.) must be redefined inline.
 * This duplication is inherent to Chrome's extension security model.
 */

export async function handleGetDetectedPapers() {
  const tab = await getActiveTab();
  if (!tab?.id) return { papers: [], type: null };

  const cached = detectedPapers.get(tab.id);
  if (cached) {
    return { papers: cached.papers, type: cached.type };
  }

  // Try to run generic detection via activeTab on the current page
  try {
    const results = await chrome.tabs.sendMessage(tab.id, { type: "EXTRACT_PAPERS" });
    if (results?.papers?.length > 0) {
      detectedPapers.set(tab.id, {
        type: results.type ?? "single",
        papers: results.papers,
        translatorName: "generic",
      });
      return { papers: results.papers, type: results.type ?? "single" };
    }
  } catch {
    // Content script not injected on this page
  }

  // Try injecting translator-runner via scripting API on the active tab
  try {
    const injectionResults = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        // Inline generic detection for pages without content script
        function getMeta(names: string[]): string | null {
          for (const name of names) {
            const el = document.querySelector(`meta[name="${name}" i], meta[property="${name}" i]`);
            if (el) {
              const content = el.getAttribute("content");
              if (content?.trim()) return content.trim();
            }
          }
          return null;
        }

        function getAllMeta(name: string): string[] {
          return Array.from(document.querySelectorAll(`meta[name="${name}" i], meta[property="${name}" i]`))
            .map((el) => el.getAttribute("content")?.trim())
            .filter((c): c is string => !!c);
        }

        const NOISE_SELECTORS = [
          "nav", "footer", ".cookie-consent", ".cookie-banner", ".cookie-notice",
          ".share-tools", ".share-widget", ".social-share",
          ".author-info", ".author-notes",
          ".metrics", ".altmetric",
          ".supplementary-data", ".supplementary-materials",
          '[role="navigation"]', '[role="banner"]',
          ".sidebar", "#sidebar",
          ".advertisement", ".ad-container",
          ".related-articles", ".recommended-articles",
          ".references", "#references",
          ".footnotes", ".endnotes",
          ".bibliography", ".Footnotes", ".Tail",
          ".RelatedContent", ".ReferencedArticles",
          ".ListArticles", ".Copyright",
          "figure", ".figure", ".table-wrap",
        ].join(", ");

        const BLOCK_TAGS = new Set([
          "div", "section", "article", "aside", "blockquote",
          "table", "figure", "ul", "ol", "pre", "form",
          "header", "footer", "nav", "main",
        ]);

        const CITATION_RE = /\[(\d+(?:[,\s]*\d+)*(?:\s*[-–]\s*\d+)?)\]/g;
        const SUPERSCRIPT_RE = /[⁰¹²³⁴⁵⁶⁷⁸⁹]+/g;

        function cleanPara(text: string): string {
          return text
            .replace(CITATION_RE, "")
            .replace(SUPERSCRIPT_RE, "")
            .replace(/[\u00a0\u200b\u200c\u200d\ufeff]/g, " ")
            .replace(/ ([.,;:!?])/g, "$1")
            .replace(/  +/g, " ")
            .trim();
        }

        function isLeafTextBlock(el: Element): boolean {
          for (const child of el.children) {
            if (BLOCK_TAGS.has(child.tagName.toLowerCase())) return false;
          }
          return true;
        }

        function inlineExtractStructured(): { full_text: string | undefined; structured_text: any } {
          const containerSelectors = [
            ".jig-ncbiinpagenav .tsec",
            "#body .section",
            "article .c-article-body",
            "article .article-body",
            "#article-body",
            ".article-section__content",
            ".article-content",
            ".Body",
            '[role="main"]',
            "article",
            "main",
          ];

          const hostname = location.hostname ?? "";
          let extractionMethod = "generic";
          if (hostname.includes("ncbi.nlm.nih.gov")) extractionMethod = "pmc";
          else if (hostname.includes("sciencedirect.com")) extractionMethod = "elsevier";
          else if (hostname.includes("nature.com") || hostname.includes("springer.com")) extractionMethod = "springer";
          else if (hostname.includes("wiley.com")) extractionMethod = "wiley";

          for (const sel of containerSelectors) {
            const containers = document.querySelectorAll(sel);
            if (containers.length === 0) continue;

            const wrapper = document.createElement("div");
            for (const c of containers) {
              wrapper.appendChild(c.cloneNode(true));
            }
            const noiseEls = wrapper.querySelectorAll(NOISE_SELECTORS);
            for (const el of noiseEls) el.remove();

            const sections: { heading?: string; paragraphs: string[] }[] = [];
            let cur: { heading?: string; paragraphs: string[] } = { paragraphs: [] };

            const elements = wrapper.querySelectorAll("h2, h3, h4, p, div");
            for (const el of elements) {
              const tag = el.tagName.toLowerCase();
              if (tag === "h2" || tag === "h3" || tag === "h4") {
                if (cur.paragraphs.length > 0) sections.push(cur);
                const h = el.textContent?.trim() ?? "";
                cur = { heading: h || undefined, paragraphs: [] };
              } else {
                if (tag === "div" && !isLeafTextBlock(el)) continue;
                const raw = el.textContent?.trim();
                if (raw && raw.length > 30) {
                  const cleaned = cleanPara(raw);
                  if (cleaned.length > 30) cur.paragraphs.push(cleaned);
                }
              }
            }
            if (cur.paragraphs.length > 0) sections.push(cur);

            const allParas: string[] = [];
            for (const s of sections) for (const p of s.paragraphs) allParas.push(p);
            const fullText = allParas.join("\n\n");

            if (fullText.length >= 200) {
              return {
                full_text: fullText,
                structured_text: { sections, extraction_method: extractionMethod, source_hostname: hostname },
              };
            }
          }
          return { full_text: undefined, structured_text: undefined };
        }

        function extractAbstractFromDom(): string | null {
          const selectors = [
            ".Abstracts .abstract.author",
            ".abstract-content",
            '[class*="abstract"] p',
            "#abstract p",
            ".hlFld-Abstract p",
            ".abstractSection",
          ];
          for (const sel of selectors) {
            const els = document.querySelectorAll(sel);
            if (els.length === 0) continue;
            const texts: string[] = [];
            for (const el of els) {
              const text = el.textContent?.trim();
              if (text && text.length > 30) texts.push(cleanPara(text));
            }
            const combined = texts.join(" ");
            if (combined.length > 100) return combined.replace(/^Abstract\s*/i, "");
          }
          return null;
        }

        const title = getMeta(["citation_title", "DC.Title", "DC.title", "og:title"]);
        if (!title) return { papers: [], type: null };

        const authors = getAllMeta("citation_author");
        const doi = getMeta(["citation_doi", "DC.Identifier"]) ?? undefined;
        let abstract = getMeta(["citation_abstract", "DC.Description", "og:description"]) ?? undefined;
        if (!abstract || abstract.length < 200) {
          const domAbstract = extractAbstractFromDom();
          if (domAbstract && domAbstract.length > (abstract?.length ?? 0)) {
            abstract = domAbstract;
          }
        }
        const journal = getMeta(["citation_journal_title", "DC.Source"]) ?? undefined;
        const dateStr = getMeta(["citation_date", "citation_publication_date", "DC.Date"]);
        const year = dateStr ? parseInt(dateStr.match(/(\d{4})/)?.[1] ?? "", 10) || undefined : undefined;
        const pdf_url = getMeta(["citation_pdf_url"]) ?? undefined;

        // Additional metadata fields
        const volume = getMeta(["citation_volume", "PRISM.volume"]) ?? undefined;
        const issue = getMeta(["citation_issue", "PRISM.number"]) ?? undefined;
        const firstPage = getMeta(["citation_firstpage"]);
        const lastPage = getMeta(["citation_lastpage"]);
        const pages = firstPage ? (lastPage ? `${firstPage}-${lastPage}` : firstPage) : undefined;
        const pmid = getMeta(["citation_pmid"]) ?? undefined;
        const issn = getMeta(["citation_issn", "PRISM.issn", "PRISM.eIssn"]) ?? undefined;
        const publisher = getMeta(["citation_publisher", "DC.Publisher"]) ?? undefined;
        const language = getMeta(["citation_language", "DC.Language"]) ?? undefined;
        let keywords: string[] | undefined;
        const keywordStr = getMeta(["citation_keywords"]);
        if (keywordStr) {
          keywords = keywordStr.split(",").map((k: string) => k.trim()).filter(Boolean);
        }
        if (!keywords?.length) {
          const dcSubjects = getAllMeta("DC.Subject");
          if (dcSubjects.length) keywords = dcSubjects;
        }

        const { full_text, structured_text } = inlineExtractStructured();

        return {
          papers: [{
            title,
            authors: authors.length ? authors : undefined,
            year,
            doi,
            abstract,
            journal,
            url: location.href,
            pdf_url,
            full_text,
            structured_text,
            volume,
            issue,
            pages,
            pmid,
            issn,
            publisher,
            keywords: keywords?.length ? keywords : undefined,
            language,
          }],
          type: "single",
        };
      },
    });

    const result = injectionResults?.[0]?.result as
      | { papers: PaperMetadata[]; type: "single" | "multiple" | null }
      | undefined;
    if (result && result.papers && result.papers.length > 0) {
      const detectedType: "single" | "multiple" = result.type === "multiple" ? "multiple" : "single";
      detectedPapers.set(tab.id, {
        type: detectedType,
        papers: result.papers,
        translatorName: "generic-injected",
      });

      // Set popup mode and badge for this tab
      await chrome.action.setPopup({ tabId: tab.id, popup: "popup/popup.html" });
      await updateBadge(tab.id, detectedType, result.papers);

      return { papers: result.papers, type: detectedType };
    }
  } catch {
    // Injection not allowed on this page
  }

  return { papers: [], type: null };
}
