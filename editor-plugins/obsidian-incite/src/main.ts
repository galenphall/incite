import { Editor, Notice, Plugin, WorkspaceLeaf } from "obsidian";
import { InCiteClient } from "./api-client";
import { CitationWatcher } from "./citation-watcher";
import { ObsidianCitationStorage } from "./citation-storage";
import { FrontmatterCitationStorage } from "./frontmatter-citation-storage";
import { extractContext } from "./context-extractor";
import { InCiteSettingTab } from "./settings";
import { InCiteSidebarView, VIEW_TYPE_INCITE } from "./sidebar-view";
import type { InCiteSettings, Recommendation } from "./types";
import {
	DEFAULT_SETTINGS,
	getActiveUrl,
	formatMultiCitation,
	CitationTracker,
	exportBibTeX,
	exportRIS,
	exportFormattedText,
} from "./types";

export default class InCitePlugin extends Plugin {
	settings: InCiteSettings = DEFAULT_SETTINGS;
	client: InCiteClient = new InCiteClient({
		apiMode: DEFAULT_SETTINGS.apiMode,
		cloudUrl: DEFAULT_SETTINGS.cloudUrl,
		localUrl: DEFAULT_SETTINGS.localUrl,
		apiToken: DEFAULT_SETTINGS.apiToken,
	});
	private watcher: CitationWatcher | null = null;
	private lastEditor: Editor | null = null;
	private tracker: CitationTracker | null = null;
	private citationStorage: FrontmatterCitationStorage | null = null;
	private legacyStorage: ObsidianCitationStorage | null = null;

	async onload(): Promise<void> {
		await this.loadSettings();
		this.citationStorage = new FrontmatterCitationStorage(this.app);
		this.legacyStorage = new ObsidianCitationStorage(this);

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

		// Fetch collections for cloud mode
		if (this.settings.apiMode === "cloud") {
			this.fetchCollections();
		}
	}

	onunload(): void {
		this.stopWatcher();
	}

	async loadSettings(): Promise<void> {
		const loaded = await this.loadData();

		// Migration: old apiUrl → new apiMode/localUrl/cloudUrl
		if (loaded?.apiUrl && !loaded?.apiMode) {
			loaded.localUrl = loaded.apiUrl;
			loaded.apiMode = "local";
			delete loaded.apiUrl;
		}

		this.settings = Object.assign({}, DEFAULT_SETTINGS, loaded);
		this.client.updateConfig({
			apiMode: this.settings.apiMode,
			cloudUrl: this.settings.cloudUrl,
			localUrl: this.settings.localUrl,
			apiToken: this.settings.apiToken,
		});
	}

	async saveSettings(): Promise<void> {
		await this.saveData(this.settings);
		this.client.updateConfig({
			apiMode: this.settings.apiMode,
			cloudUrl: this.settings.cloudUrl,
			localUrl: this.settings.localUrl,
			apiToken: this.settings.apiToken,
		});

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

		// Refresh collections when switching to cloud mode
		if (this.settings.apiMode === "cloud") {
			this.fetchCollections();
		}
	}

	/** Initialize the citation tracker for the currently active file. */
	private async initTrackerForActiveFile(): Promise<void> {
		if (!this.citationStorage) return;
		const file = this.app.workspace.getActiveFile();
		const docKey = file?.path ?? "__no_file__";
		this.tracker = new CitationTracker(this.citationStorage, docKey);
		await this.tracker.load();

		// Migrate from legacy plugin-data storage to frontmatter
		if (this.tracker.count === 0 && this.legacyStorage) {
			const legacy = await this.legacyStorage.load(docKey);
			if (legacy.length > 0) {
				await this.citationStorage.save(docKey, legacy);
				await this.tracker.load();
			}
		}

		this.refreshBibliography();
	}

	/** Push current tracked citations to the sidebar view. */
	private refreshBibliography(): void {
		const view = this.getSidebarView();
		if (!view || !this.tracker) return;
		view.setBibliography(
			this.tracker.getAll(),
			(paperId) => this.removeCitation(paperId),
			(format) => this.exportBibliography(format),
			() => this.insertBibliography()
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

	/** Insert APA-formatted bibliography at the current cursor position. */
	private insertBibliography(): void {
		const editor = this.lastEditor ?? this.app.workspace.activeEditor?.editor ?? null;
		if (!editor) {
			new Notice("No active editor to insert bibliography into.");
			return;
		}
		if (!this.tracker || this.tracker.count === 0) {
			new Notice("No citations to insert.");
			return;
		}

		const text = exportFormattedText(this.tracker.getAll());
		const cursor = editor.getCursor();
		editor.replaceRange(text, cursor);

		const newOffset = editor.posToOffset(cursor) + text.length;
		editor.setCursor(editor.offsetToPos(newOffset));

		new Notice(`Inserted bibliography (${this.tracker.count} citations).`);
	}

	/** Fetch collections from cloud and update sidebar dropdown. */
	async fetchCollections(): Promise<void> {
		try {
			const collections = await this.client.getCollections();
			const view = this.getSidebarView();
			if (view) {
				view.setCollections(collections, (collectionId) => {
					this.settings.collectionId = collectionId;
					this.saveSettings();
				});
			}
		} catch {
			// Collections are optional — silently ignore
		}
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
				cursorSentenceIndex,
				this.settings.collectionId,
			);
			if (view) {
				view.setResults(response.recommendations);
				this.refreshBibliography();
			}
		} catch (err) {
			const message =
				err instanceof Error ? err.message : "Failed to connect";
			if (this.settings.apiMode === "cloud") {
				if (view) {
					view.setError(
						`Could not reach inCite cloud. Check your API token. (${message})`
					);
				}
				new Notice("inCite: Could not reach cloud. Check your API token.");
			} else {
				const url = getActiveUrl(this.settings);
				if (view) {
					view.setError(
						`Could not reach inCite at ${url}. ` +
							`Is 'incite serve' running? (${message})`
					);
				}
				new Notice("inCite: Server not reachable. Run 'incite serve'.");
			}
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
