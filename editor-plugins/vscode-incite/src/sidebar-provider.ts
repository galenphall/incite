import * as vscode from "vscode";
import type { InCiteSettings, Recommendation } from "./types";

/**
 * WebviewViewProvider that renders the inCite sidebar.
 *
 * All HTML/CSS/JS is embedded inline for simplicity (v1).
 * Communication with the extension host uses postMessage.
 */
export class SidebarProvider implements vscode.WebviewViewProvider {
	public static readonly viewType = "incite.sidebar";

	private view?: vscode.WebviewView;
	private settings: InCiteSettings;
	private onInsert: (rec: Recommendation) => void;

	constructor(
		private readonly extensionUri: vscode.Uri,
		settings: InCiteSettings,
		onInsert: (rec: Recommendation) => void
	) {
		this.settings = settings;
		this.onInsert = onInsert;
	}

	resolveWebviewView(
		webviewView: vscode.WebviewView,
		_context: vscode.WebviewViewResolveContext,
		_token: vscode.CancellationToken
	): void {
		this.view = webviewView;

		webviewView.webview.options = {
			enableScripts: true,
		};

		webviewView.webview.html = this.getHtml();

		// Handle messages from the webview
		webviewView.webview.onDidReceiveMessage((message) => {
			switch (message.type) {
				case "insert": {
					const rec = message.recommendation as Recommendation;
					this.onInsert(rec);
					break;
				}
			}
		});
	}

	/** Update settings reference. */
	updateSettings(settings: InCiteSettings): void {
		this.settings = settings;
	}

	/** Show loading state. */
	setLoading(): void {
		this.postMessage({ type: "loading" });
	}

	/** Show error message. */
	setError(message: string): void {
		this.postMessage({ type: "error", message });
	}

	/** Display recommendation results. */
	setResults(results: Recommendation[]): void {
		this.postMessage({
			type: "results",
			recommendations: results,
			showParagraphs: this.settings.showParagraphs,
		});
	}

	private postMessage(message: unknown): void {
		if (this.view) {
			this.view.webview.postMessage(message);
		}
	}

	private getHtml(): string {
		return /*html*/ `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
	:root {
		--bg: var(--vscode-sideBar-background);
		--fg: var(--vscode-sideBar-foreground);
		--border: var(--vscode-panel-border);
		--link: var(--vscode-textLink-foreground);
		--muted: var(--vscode-descriptionForeground);
		--badge-bg: var(--vscode-badge-background);
		--badge-fg: var(--vscode-badge-foreground);
		--btn-bg: var(--vscode-button-background);
		--btn-fg: var(--vscode-button-foreground);
		--btn-hover: var(--vscode-button-hoverBackground);
		--input-bg: var(--vscode-input-background);
		--blockquote-bg: var(--vscode-textBlockQuote-background);
		--blockquote-border: var(--vscode-textBlockQuote-border);
	}
	* { box-sizing: border-box; margin: 0; padding: 0; }
	body {
		font-family: var(--vscode-font-family);
		font-size: var(--vscode-font-size);
		color: var(--fg);
		background: var(--bg);
		padding: 8px;
	}
	.header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 8px;
		padding-bottom: 6px;
		border-bottom: 1px solid var(--border);
	}
	.header h3 {
		font-size: 13px;
		font-weight: 600;
	}
	.status {
		font-size: 11px;
		color: var(--muted);
	}
	.message {
		text-align: center;
		padding: 24px 12px;
		color: var(--muted);
		font-size: 12px;
		line-height: 1.5;
	}
	.error {
		color: var(--vscode-errorForeground);
	}
	.spinner {
		display: inline-block;
		width: 16px;
		height: 16px;
		border: 2px solid var(--muted);
		border-top-color: var(--link);
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
		margin-bottom: 8px;
	}
	@keyframes spin { to { transform: rotate(360deg); } }
	.result {
		padding: 8px;
		margin-bottom: 6px;
		border: 1px solid var(--border);
		border-radius: 4px;
		background: var(--input-bg);
	}
	.result:hover {
		border-color: var(--link);
	}
	.result-rank {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 4px;
	}
	.rank-number {
		font-weight: 600;
		font-size: 11px;
		color: var(--muted);
	}
	.score {
		font-size: 10px;
		color: var(--badge-fg);
		background: var(--badge-bg);
		padding: 1px 5px;
		border-radius: 8px;
	}
	.result-title {
		font-weight: 600;
		font-size: 12px;
		line-height: 1.35;
		margin-bottom: 3px;
	}
	.result-meta {
		font-size: 11px;
		color: var(--muted);
		margin-bottom: 4px;
	}
	.paragraph {
		font-size: 11px;
		color: var(--muted);
		border-left: 3px solid var(--blockquote-border);
		background: var(--blockquote-bg);
		padding: 4px 8px;
		margin: 4px 0;
		line-height: 1.4;
	}
	.paragraph-secondary {
		opacity: 0.75;
		font-size: 10px;
		margin-top: 2px;
	}
	.evidence-score {
		font-size: 10px;
		font-weight: 600;
		color: var(--link-foreground);
		margin-right: 4px;
	}
	.actions {
		display: flex;
		gap: 6px;
		margin-top: 4px;
	}
	.insert-btn {
		font-size: 11px;
		padding: 2px 8px;
		border: none;
		border-radius: 3px;
		background: var(--btn-bg);
		color: var(--btn-fg);
		cursor: pointer;
	}
	.insert-btn:hover {
		background: var(--btn-hover);
	}
</style>
</head>
<body>
	<div class="header">
		<h3>inCite</h3>
		<span class="status" id="status"></span>
	</div>
	<div id="content">
		<div class="message">
			No results yet.<br>
			Place your cursor and press <kbd>Cmd/Ctrl+Shift+C</kbd>.
		</div>
	</div>

<script>
	const vscode = acquireVsCodeApi();
	const content = document.getElementById("content");
	const status = document.getElementById("status");

	window.addEventListener("message", (event) => {
		const msg = event.data;
		switch (msg.type) {
			case "loading":
				content.innerHTML = '<div class="message"><div class="spinner"></div><br>Searching...</div>';
				status.textContent = "";
				break;

			case "error":
				content.innerHTML = '<div class="message error">' + escapeHtml(msg.message) + '</div>';
				status.textContent = "";
				break;

			case "results":
				renderResults(msg.recommendations, msg.showParagraphs);
				break;
		}
	});

	function renderResults(recs, showParagraphs) {
		if (!recs || recs.length === 0) {
			content.innerHTML = '<div class="message">No matching papers found.</div>';
			status.textContent = "";
			return;
		}

		status.textContent = recs.length + " results";
		let html = "";

		for (const rec of recs) {
			html += '<div class="result">';

			// Rank + score
			html += '<div class="result-rank">';
			html += '<span class="rank-number">' + rec.rank + '.</span>';
			html += '<span class="score">' + rec.score.toFixed(3) + '</span>';
			html += '</div>';

			// Title
			html += '<div class="result-title">' + escapeHtml(rec.title) + '</div>';

			// Authors + year
			const meta = [];
			if (rec.authors && rec.authors.length > 0) {
				const names = rec.authors.slice(0, 3).join(", ");
				meta.push(rec.authors.length > 3 ? names + " et al." : names);
			}
			if (rec.year) {
				meta.push("(" + rec.year + ")");
			}
			if (meta.length > 0) {
				html += '<div class="result-meta">' + escapeHtml(meta.join(" ")) + '</div>';
			}

			// Matched paragraph(s)
			if (showParagraphs) {
				if (rec.matched_paragraphs?.length) {
					rec.matched_paragraphs.forEach((snippet: any, idx: number) => {
						const text = snippet.text.length > 300
							? snippet.text.slice(0, 300) + "..."
							: snippet.text;
						const cls = idx === 0 ? "paragraph" : "paragraph paragraph-secondary";
						const badge = snippet.score != null
							? '<span class="evidence-score">' + Math.round(snippet.score * 100) + '%</span> '
							: '';
						html += '<div class="' + cls + '">' + badge + escapeHtml(text) + '</div>';
					});
				} else if (rec.matched_paragraph) {
					const text = rec.matched_paragraph.length > 300
						? rec.matched_paragraph.slice(0, 300) + "..."
						: rec.matched_paragraph;
					html += '<div class="paragraph">' + escapeHtml(text) + '</div>';
				}
			}

			// Insert button
			html += '<div class="actions">';
			const recJson = escapeAttr(JSON.stringify(rec));
			html += "<button class='insert-btn' data-rec='" + recJson + "'>Insert</button>";
			html += '</div>';

			html += '</div>';
		}

		content.innerHTML = html;

		// Attach click handlers
		content.querySelectorAll(".insert-btn").forEach((btn) => {
			btn.addEventListener("click", () => {
				const rec = JSON.parse(btn.getAttribute("data-rec"));
				vscode.postMessage({ type: "insert", recommendation: rec });
			});
		});
	}

	function escapeHtml(text) {
		const div = document.createElement("div");
		div.textContent = text;
		return div.innerHTML;
	}

	function escapeAttr(text) {
		return text.replace(/&/g, "&amp;").replace(/'/g, "&#39;").replace(/"/g, "&quot;");
	}
</script>
</body>
</html>`;
	}
}
