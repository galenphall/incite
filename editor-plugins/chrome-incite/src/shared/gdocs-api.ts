/** Google Docs REST API client for Chrome extension OAuth flow. */

// ---------------------------------------------------------------------------
// Type definitions (simplified — only fields we use)
// ---------------------------------------------------------------------------

export interface DocsDocument {
  documentId: string;
  title: string;
  body: DocsBody;
  namedRanges?: Record<string, DocsNamedRanges>;
}

export interface DocsBody {
  content: DocsStructuralElement[];
}

export interface DocsStructuralElement {
  startIndex: number;
  endIndex: number;
  paragraph?: DocsParagraph;
  table?: DocsTable;
  sectionBreak?: object;
}

export interface DocsParagraph {
  elements: DocsParagraphElement[];
  paragraphStyle?: { namedStyleType?: string };
}

export interface DocsParagraphElement {
  startIndex: number;
  endIndex: number;
  textRun?: { content: string; textStyle?: DocsTextStyle };
}

export interface DocsTextStyle {
  bold?: boolean;
  italic?: boolean;
  link?: { url: string };
}

export interface DocsTable {
  rows: number;
  columns: number;
  tableRows: DocsTableRow[];
}

export interface DocsTableRow {
  tableCells: DocsTableCell[];
}

export interface DocsTableCell {
  content: DocsStructuralElement[];
}

export interface DocsNamedRanges {
  name: string;
  namedRanges: DocsNamedRange[];
}

export interface DocsNamedRange {
  namedRangeId: string;
  name: string;
  ranges: DocsRange[];
}

export interface DocsRange {
  startIndex: number;
  endIndex: number;
  segmentId?: string;
}

export interface BatchUpdateResponse {
  documentId: string;
  replies: object[];
  writeControl?: { requiredRevisionId?: string };
}

export type DocsRequest =
  | { insertText: { location: { index: number }; text: string } }
  | { updateTextStyle: { range: { startIndex: number; endIndex: number }; textStyle: DocsTextStyle; fields: string } }
  | { createNamedRange: { name: string; range: { startIndex: number; endIndex: number } } }
  | { deleteNamedRange: { namedRangeId: string } }
  | { deleteContentRange: { range: { startIndex: number; endIndex: number } } };

// ---------------------------------------------------------------------------
// API client
// ---------------------------------------------------------------------------

const DOCS_BASE = "https://docs.googleapis.com/v1/documents";
const OAUTH_SCOPES = "https://www.googleapis.com/auth/documents";

export class GoogleDocsAPI {
  private token: string | null = null;

  /**
   * Authenticate via chrome.identity.launchWebAuthFlow (works in Chrome, Brave, Edge, etc.).
   * Uses the implicit grant flow to get an access token directly.
   */
  async authenticate(interactive = true): Promise<void> {
    // Read client_id from manifest
    const manifest = chrome.runtime.getManifest();
    const clientId = manifest.oauth2?.client_id;
    if (!clientId) {
      throw new Error("No oauth2.client_id in manifest.json");
    }

    const redirectUrl = chrome.identity.getRedirectURL();
    const authUrl = new URL("https://accounts.google.com/o/oauth2/v2/auth");
    authUrl.searchParams.set("client_id", clientId);
    authUrl.searchParams.set("redirect_uri", redirectUrl);
    authUrl.searchParams.set("response_type", "token");
    authUrl.searchParams.set("scope", OAUTH_SCOPES);
    authUrl.searchParams.set("prompt", "consent");

    const responseUrl = await chrome.identity.launchWebAuthFlow({
      url: authUrl.toString(),
      interactive,
    });

    if (!responseUrl) {
      throw new Error("OAuth flow returned no response");
    }

    // Extract access_token from the URL fragment
    const fragment = new URL(responseUrl).hash.substring(1);
    const params = new URLSearchParams(fragment);
    const accessToken = params.get("access_token");
    if (!accessToken) {
      throw new Error("No access_token in OAuth response");
    }
    this.token = accessToken;
  }

  async refreshToken(): Promise<void> {
    this.token = null;
    await this.authenticate();
  }

  private async request<T>(method: string, url: string, body?: unknown): Promise<T> {
    if (!this.token) {
      await this.authenticate();
    }

    const doFetch = async (): Promise<Response> => {
      const init: RequestInit = {
        method,
        headers: {
          Authorization: `Bearer ${this.token}`,
          "Content-Type": "application/json",
        },
      };
      if (body !== undefined) {
        init.body = JSON.stringify(body);
      }
      return fetch(url, init);
    };

    let resp = await doFetch();

    if (resp.status === 401) {
      await this.refreshToken();
      resp = await doFetch();
    }

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Google Docs API ${method} ${url} failed (${resp.status}): ${text}`);
    }

    return resp.json() as Promise<T>;
  }

  async getDocument(docId: string): Promise<DocsDocument> {
    return this.request("GET", `${DOCS_BASE}/${docId}`);
  }

  async batchUpdate(docId: string, requests: DocsRequest[]): Promise<BatchUpdateResponse> {
    return this.request("POST", `${DOCS_BASE}/${docId}:batchUpdate`, { requests });
  }

  // -----------------------------------------------------------------------
  // Convenience methods
  // -----------------------------------------------------------------------

  async insertTextAt(docId: string, index: number, text: string): Promise<void> {
    await this.batchUpdate(docId, [{ insertText: { location: { index }, text } }]);
  }

  async insertLinkedTextAt(docId: string, index: number, text: string, url: string): Promise<void> {
    await this.batchUpdate(docId, [
      { insertText: { location: { index }, text } },
      {
        updateTextStyle: {
          range: { startIndex: index, endIndex: index + text.length },
          textStyle: { link: { url } },
          fields: "link",
        },
      },
    ]);
  }

  async createNamedRange(docId: string, name: string, start: number, end: number): Promise<string> {
    const resp = await this.batchUpdate(docId, [
      { createNamedRange: { name, range: { startIndex: start, endIndex: end } } },
    ]);
    const reply = resp.replies[0] as { createNamedRange?: { namedRangeId: string } };
    return reply.createNamedRange?.namedRangeId ?? "";
  }

  async deleteRange(docId: string, start: number, end: number): Promise<void> {
    await this.batchUpdate(docId, [
      { deleteContentRange: { range: { startIndex: start, endIndex: end } } },
    ]);
  }

  async findNamedRanges(docId: string, prefix: string): Promise<DocsNamedRange[]> {
    const doc = await this.getDocument(docId);
    const results: DocsNamedRange[] = [];
    if (!doc.namedRanges) return results;
    for (const [name, entry] of Object.entries(doc.namedRanges)) {
      if (name.startsWith(prefix)) {
        results.push(...entry.namedRanges);
      }
    }
    return results;
  }

  extractDocumentText(doc: DocsDocument): { text: string; indexMap: { docIndex: number; length: number }[] } {
    const parts: { text: string; docIndex: number }[] = [];

    const extractFromElements = (elements: DocsStructuralElement[]): void => {
      for (const element of elements) {
        if (element.paragraph) {
          for (const pe of element.paragraph.elements) {
            if (pe.textRun?.content) {
              parts.push({ text: pe.textRun.content, docIndex: pe.startIndex });
            }
          }
        }
        if (element.table) {
          for (const row of element.table.tableRows) {
            for (const cell of row.tableCells) {
              extractFromElements(cell.content);
            }
          }
        }
      }
    };

    extractFromElements(doc.body.content);

    const text = parts.map((p) => p.text).join("");
    const indexMap = parts.map((p) => ({ docIndex: p.docIndex, length: p.text.length }));
    return { text, indexMap };
  }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

export function extractDocId(url: string): string | null {
  const match = url.match(/\/document\/d\/([a-zA-Z0-9_-]+)/);
  return match?.[1] ?? null;
}
