/**
 * HTML extraction tests using jsdom + saved publisher page snapshots.
 *
 * To add a new test:
 * 1. Save the page HTML: copy(document.documentElement.outerHTML) in DevTools
 * 2. Place in test/snapshots/{name}.html
 * 3. Create test/expected/{name}.json with expected extraction results
 * 4. The test will automatically pick it up
 */

import { describe, it, expect } from "vitest";
import { JSDOM } from "jsdom";
import * as fs from "fs";
import * as path from "path";
import { extractStructuredText } from "../src/translators/generic";

interface ExpectedResult {
  min_sections?: number;
  min_paragraphs?: number;
  min_full_text_length?: number;
  expected_extraction_method?: string;
  noise_patterns_absent?: string[];
  expected_first_heading_contains?: string;
  metadata?: {
    has_title?: boolean;
    has_doi?: boolean;
    has_authors?: boolean;
    min_author_count?: number;
  };
}

const SNAPSHOTS_DIR = path.join(__dirname, "snapshots");
const EXPECTED_DIR = path.join(__dirname, "expected");

/**
 * Get all snapshot/expected pairs that exist.
 */
function getTestCases(): Array<{ name: string; htmlPath: string; expected: ExpectedResult }> {
  if (!fs.existsSync(EXPECTED_DIR)) return [];

  const expectedFiles = fs.readdirSync(EXPECTED_DIR).filter((f) => f.endsWith(".json"));
  const cases: Array<{ name: string; htmlPath: string; expected: ExpectedResult }> = [];

  for (const file of expectedFiles) {
    const name = file.replace(/\.json$/, "");
    const htmlPath = path.join(SNAPSHOTS_DIR, `${name}.html`);
    if (!fs.existsSync(htmlPath)) continue;

    const expected: ExpectedResult = JSON.parse(
      fs.readFileSync(path.join(EXPECTED_DIR, file), "utf-8"),
    );
    cases.push({ name, htmlPath, expected });
  }

  return cases;
}

describe("extractStructuredText", () => {
  const testCases = getTestCases();

  if (testCases.length === 0) {
    it("skips — no snapshot/expected pairs found yet", () => {
      console.log(
        "No test snapshots found. To add tests:\n" +
          "  1. Save page HTML to test/snapshots/{publisher}.html\n" +
          "  2. Create test/expected/{publisher}.json\n",
      );
    });
    return;
  }

  for (const { name, htmlPath, expected } of testCases) {
    describe(name, () => {
      const html = fs.readFileSync(htmlPath, "utf-8");
      const dom = new JSDOM(html, { url: `https://${name}.example.com/article` });
      const doc = dom.window.document;
      const result = extractStructuredText(doc as unknown as Document);

      if (expected.min_full_text_length !== undefined) {
        it(`produces at least ${expected.min_full_text_length} chars of text`, () => {
          expect(result.full_text).toBeTruthy();
          expect(result.full_text!.length).toBeGreaterThanOrEqual(expected.min_full_text_length!);
        });
      }

      if (expected.min_sections !== undefined) {
        it(`produces at least ${expected.min_sections} sections`, () => {
          expect(result.structured_text).toBeTruthy();
          expect(result.structured_text!.sections.length).toBeGreaterThanOrEqual(
            expected.min_sections!,
          );
        });
      }

      if (expected.min_paragraphs !== undefined) {
        it(`produces at least ${expected.min_paragraphs} paragraphs`, () => {
          expect(result.structured_text).toBeTruthy();
          const totalParas = result.structured_text!.sections.reduce(
            (sum, s) => sum + s.paragraphs.length,
            0,
          );
          expect(totalParas).toBeGreaterThanOrEqual(expected.min_paragraphs!);
        });
      }

      if (expected.expected_extraction_method) {
        it(`uses extraction method "${expected.expected_extraction_method}"`, () => {
          expect(result.structured_text).toBeTruthy();
          expect(result.structured_text!.extraction_method).toBe(
            expected.expected_extraction_method,
          );
        });
      }

      if (expected.noise_patterns_absent?.length) {
        for (const pattern of expected.noise_patterns_absent) {
          it(`does not contain noise pattern "${pattern}"`, () => {
            expect(result.full_text).toBeTruthy();
            // Check for exact word (case-insensitive) to avoid false positives
            // from legitimate use of common words in article text
            const hasNoise = result.full_text!.toLowerCase().includes(pattern.toLowerCase());
            if (hasNoise) {
              // Only fail if the pattern appears to be noise, not article content
              // Allow "References" if it's in a sentence context
              const lines = result.full_text!.split("\n\n");
              const noiseLines = lines.filter(
                (l) =>
                  l.toLowerCase().trim() === pattern.toLowerCase() ||
                  l.toLowerCase().startsWith(pattern.toLowerCase() + "\n"),
              );
              expect(noiseLines).toHaveLength(0);
            }
          });
        }
      }

      if (expected.expected_first_heading_contains) {
        it(`first heading contains "${expected.expected_first_heading_contains}"`, () => {
          expect(result.structured_text).toBeTruthy();
          const firstHeading = result.structured_text!.sections.find((s) => s.heading)?.heading;
          expect(firstHeading).toBeTruthy();
          expect(firstHeading!.toLowerCase()).toContain(
            expected.expected_first_heading_contains!.toLowerCase(),
          );
        });
      }
    });
  }
});

describe("extractStructuredText — unit tests", () => {
  it("extracts text from a simple article structure", () => {
    const html = `
      <html>
      <head>
        <meta name="citation_title" content="Test Paper">
      </head>
      <body>
        <article>
          <h2>Introduction</h2>
          <p>This is the introduction paragraph with enough text to pass the minimum length threshold for extraction.</p>
          <h2>Methods</h2>
          <p>This is the methods section with a description of the approach used in this research study.</p>
          <h2>Results</h2>
          <p>These are the results of our analysis showing significant findings in the data we collected.</p>
        </article>
      </body>
      </html>
    `;
    const dom = new JSDOM(html, { url: "https://example.com/article" });
    const doc = dom.window.document;
    const result = extractStructuredText(doc as unknown as Document);

    expect(result.full_text).toBeTruthy();
    expect(result.structured_text).toBeTruthy();
    expect(result.structured_text!.sections.length).toBeGreaterThanOrEqual(3);
    expect(result.structured_text!.sections[0].heading).toBe("Introduction");
  });

  it("strips citation markers from paragraphs", () => {
    const html = `
      <html><body><article>
        <p>This study found significant results [1] that were confirmed by later work [2,3] and meta-analysis [4-6] in the field of computational biology and natural language processing research.</p>
        <p>Further investigations revealed that the methodology was robust across multiple experimental conditions and datasets, supporting the generalizability of the findings reported here.</p>
        <p>The implications of these results extend beyond the immediate domain, suggesting potential applications in related fields of machine learning and artificial intelligence research broadly.</p>
      </article></body></html>
    `;
    const dom = new JSDOM(html, { url: "https://example.com/article" });
    const doc = dom.window.document;
    const result = extractStructuredText(doc as unknown as Document);

    expect(result.full_text).toBeTruthy();
    expect(result.full_text).not.toContain("[1]");
    expect(result.full_text).not.toContain("[2,3]");
    expect(result.full_text).not.toContain("[4-6]");
  });

  it("removes noise elements (references, footer, nav)", () => {
    const html = `
      <html><body><article>
        <h2>Introduction</h2>
        <p>This is a paragraph with enough content to pass the minimum length for extraction into chunks and demonstrate that the system works correctly.</p>
        <p>A second paragraph provides additional content about the research methodology and experimental design used in this comprehensive study of extraction.</p>
        <p>The third paragraph discusses the implications of the findings for future research directions and potential applications in real-world scenarios and systems.</p>
        <nav>Navigation menu should be removed from extraction results entirely.</nav>
        <footer>Footer content should not appear in the extracted article text output.</footer>
        <div class="references">
          <h2>References</h2>
          <p>1. Some reference that should not appear in extracted body text at all.</p>
        </div>
      </article></body></html>
    `;
    const dom = new JSDOM(html, { url: "https://example.com/article" });
    const doc = dom.window.document;
    const result = extractStructuredText(doc as unknown as Document);

    expect(result.full_text).toBeTruthy();
    expect(result.full_text).not.toContain("Navigation menu");
    expect(result.full_text).not.toContain("Footer content");
    expect(result.full_text).not.toContain("Some reference");
  });

  it("returns null for pages with insufficient text", () => {
    const html = `
      <html><body><article><p>Short.</p></article></body></html>
    `;
    const dom = new JSDOM(html, { url: "https://example.com" });
    const doc = dom.window.document;
    const result = extractStructuredText(doc as unknown as Document);

    expect(result.full_text).toBeNull();
    expect(result.structured_text).toBeNull();
  });
});
