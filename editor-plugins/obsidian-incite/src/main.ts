import { Editor, Notice, Plugin, WorkspaceLeaf } from "obsidian";
import { InCiteClient } from "./api-client";
import { CitationWatcher } from "./citation-watcher";
import { ObsidianCitationStorage } from "./citation-storage";
import { extractContext } from "./context-extractor";
import { InCiteSettingTab } from "./settings";
import { InCiteSidebarView, VIEW_TYPE_INCITE } from "./sidebar-view";
import type { InCiteSettings, Recommendation } from "./types";
import {
	DEFAULT_SETTINGS,
	formatMultiCitation,
	CitationTracker,
	exportBibTeX,
	exportRIS,
	exportFormattedText,
} from "./types";

export default class InCitePlugin extends Plugin {
	settings: InCiteSettings = DEFAULT_SETTINGS;
	client: InCiteClient = new InCiteClient(DEFAULT_SETTINGS.apiUrl);
	private watcher: CitationWatcher | null = null;
	private lastEditor: Editor | null = null;
	private tracker: CitationTracker | null = null;
	private citationStorage: ObsidianCitationStorage | null = null;

	async onload(): Promise<void> {
		await this.loadSettings();
		this.citationStorage = new ObsidianCitationStorage(this);

		// Register the sidebar view
		this.registerView(VIEW_TYPE_INCITE, (leaf) => {
			return new InCiteSidebarView(
				leaf,
				this.settings,
				(rec) => this.insertCitation(rec),
				(recs) => this.insertMultiCitation(recs),
				(paperId) => this.tracker?.isTracked(paperId) ?? false
			);
		});

		// Command: Get recommendations at cursor
		this.addCommand({
			id: "recommend-at-cursor",
			name: "Get citation recommendations",
			editorCallback: (editor: Editor) => {
				this.recommendAtCursor(editor);
			},
			hotkeys: [{ modifiers: ["Mod", "Shift"], key: "c" }],
		});

		// Command: Open sidebar
		this.addCommand({
			id: "open-sidebar",
			name: "Open recommendations panel",
			callback: () => {
				this.activateSidebar();
			},
		});

		// Initialize tracker for active file
		this.app.workspace.on("active-leaf-change", () => {
			this.initTrackerForActiveFile();
		});
		await this.initTrackerForActiveFile();

		// Start citation watcher
		if (this.settings.autoDetectEnabled) {
			this.startWatcher();
		}

		// Settings tab
		this.addSettingTab(new InCiteSettingTab(this.app, this));
	}

	onunload(): void {
		this.stopWatcher();
	}

	async loadSettings(): Promise<void> {
		const loaded = await this.loadData();
		this.settings = Object.assign({}, DEFAULT_SETTINGS, loaded);
		this.client.setBaseUrl(this.settings.apiUrl);
	}

	async saveSettings(): Promise<void> {
		await this.saveData(this.settings);
		this.client.setBaseUrl(this.settings.apiUrl);

		// Update sidebar view settings
		const view = this.getSidebarView();
		if (view) {
			view.updateSettings(this.settings);
		}

		// Toggle watcher based on settings
		if (this.settings.autoDetectEnabled) {
			this.startWatcher();
		} else {
			this.stopWatcher();
		}
	}

	/** Initialize the citation tracker for the currently active file. */
	private async initTrackerForActiveFile(): Promise<void> {
		if (!this.citationStorage) return;
		const file = this.app.workspace.getActiveFile();
		const docKey = file?.path ?? "__no_file__";
		this.tracker = new CitationTracker(this.citationStorage, docKey);
		await this.tracker.load();
		this.refreshBibliography();
	}

	/** Push current tracked citations to the sidebar view. */
	private refreshBibliography(): void {
		const view = this.getSidebarView();
		if (!view || !this.tracker) return;
		view.setBibliography(
			this.tracker.getAll(),
			(paperId) => this.removeCitation(paperId),
			(format) => this.exportBibliography(format)
		);
	}

	/** Remove a citation from the tracker and refresh. */
	private async removeCitation(paperId: string): Promise<void> {
		if (!this.tracker) return;
		await this.tracker.remove(paperId);
		this.refreshBibliography();
	}

	/** Export bibliography in the given format and copy to clipboard. */
	private exportBibliography(format: string): void {
		if (!this.tracker) return;
		const citations = this.tracker.getAll();
		if (citations.length === 0) {
			new Notice("No citations to export.");
			return;
		}

		let output: string;
		switch (format) {
			case "bibtex":
				output = exportBibTeX(citations);
				break;
			case "ris":
				output = exportRIS(citations);
				break;
			case "text":
				output = exportFormattedText(citations);
				break;
			default:
				new Notice(`Unknown format: ${format}`);
				return;
		}

		navigator.clipboard.writeText(output).then(() => {
			new Notice(`${format.toUpperCase()} copied to clipboard (${citations.length} citations).`);
		});
	}

	/** Open or reveal the sidebar panel. */
	async activateSidebar(): Promise<void> {
		const existing = this.app.workspace.getLeavesOfType(VIEW_TYPE_INCITE);
		if (existing.length > 0) {
			this.app.workspace.revealLeaf(existing[0]);
			return;
		}

		const leaf = this.app.workspace.getRightLeaf(false);
		if (leaf) {
			await leaf.setViewState({
				type: VIEW_TYPE_INCITE,
				active: true,
			});
			this.app.workspace.revealLeaf(leaf);
		}
	}

	/** Get the current sidebar view instance, if open. */
	private getSidebarView(): InCiteSidebarView | null {
		const leaves = this.app.workspace.getLeavesOfType(VIEW_TYPE_INCITE);
		if (leaves.length > 0) {
			return leaves[0].view as InCiteSidebarView;
		}
		return null;
	}

	/** Extract context at cursor and fetch recommendations. */
	async recommendAtCursor(editor: Editor): Promise<void> {
		this.lastEditor = editor;
		await this.activateSidebar();
		const view = this.getSidebarView();

		// Use selection as query if text is selected, otherwise extract context
		const selection = editor.getSelection();
		let queryText: string;
		let cursorSentenceIndex: number | undefined;

		if (selection && selection.trim().length > 0) {
			queryText = selection.trim();
		} else {
			const fullText = editor.getValue();
			const cursor = editor.getCursor();
			const cursorOffset = editor.posToOffset(cursor);

			const context = extractContext(
				fullText,
				cursorOffset,
				this.settings.contextSentences
			);
			queryText = context.text;
			cursorSentenceIndex = context.cursorSentenceIndex;
		}

		if (!queryText.trim()) {
			if (view) view.setError("No text around cursor to use as context.");
			return;
		}

		if (view) view.setLoading();

		try {
			const response = await this.client.recommend(
				queryText,
				this.settings.k,
				this.settings.authorBoost,
				cursorSentenceIndex
			);
			if (view) {
				view.setResults(response.recommendations);
				this.refreshBibliography();
			}
		} catch (err) {
			const message =
				err instanceof Error ? err.message : "Failed to connect";
			if (view) {
				view.setError(
					`Could not reach inCite server at ${this.settings.apiUrl}. ` +
						`Is 'incite serve' running? (${message})`
				);
			}
			new Notice("inCite: Server not reachable. Run 'incite serve'.");
		}
	}

	/** Insert a formatted citation at the current cursor position. */
	insertCitation(rec: Recommendation): void {
		const editor = this.lastEditor ?? this.app.workspace.activeEditor?.editor ?? null;
		if (!editor) {
			new Notice("No active editor to insert citation into.");
			return;
		}

		// Extract last name of first author
		let firstAuthor = "Unknown";
		if (rec.authors && rec.authors.length > 0) {
			const parts = rec.authors[0].trim().split(/\s+/);
			firstAuthor = parts[parts.length - 1];
			if (rec.authors.length > 1) {
				firstAuthor += " et al.";
			}
		}

		const fallbackUri = `zotero://select/items/0_${rec.paper_id}`;
		const citation = this.settings.insertFormat
			.replace("{bibtex_key}", rec.bibtex_key || rec.paper_id)
			.replace("{paper_id}", rec.paper_id)
			.replace("{first_author}", firstAuthor)
			.replace("{year}", String(rec.year ?? "n.d."))
			.replace("{title}", rec.title)
			.replace("{zotero_uri}", rec.zotero_uri || fallbackUri);

		const cursor = editor.getCursor();
		editor.replaceRange(citation, cursor);

		// Move cursor after the inserted citation
		const newOffset = editor.posToOffset(cursor) + citation.length;
		editor.setCursor(editor.offsetToPos(newOffset));

		// Track the citation
		if (this.tracker) {
			this.tracker.track([rec]).then(() => this.refreshBibliography());
		}

		new Notice(`Inserted: ${citation}`);
	}

	/** Insert multiple citations as a grouped reference at the cursor. */
	insertMultiCitation(recs: Recommendation[]): void {
		const editor = this.lastEditor ?? this.app.workspace.activeEditor?.editor ?? null;
		if (!editor) {
			new Notice("No active editor to insert citations into.");
			return;
		}

		const citation = formatMultiCitation(recs, this.settings.insertFormat);
		const cursor = editor.getCursor();
		editor.replaceRange(citation, cursor);

		const newOffset = editor.posToOffset(cursor) + citation.length;
		editor.setCursor(editor.offsetToPos(newOffset));

		// Track all citations
		if (this.tracker) {
			this.tracker.track(recs).then(() => this.refreshBibliography());
		}

		new Notice(`Inserted ${recs.length} citations.`);
	}

	/** Start the citation pattern watcher. */
	startWatcher(): void {
		if (this.watcher) return;
		this.watcher = new CitationWatcher(
			this.app,
			this.settings,
			(editor) => this.recommendAtCursor(editor)
		);
		this.watcher.start(this);
	}

	/** Stop the citation pattern watcher. */
	stopWatcher(): void {
		if (this.watcher) {
			this.watcher.stop();
			this.watcher = null;
		}
	}
}
