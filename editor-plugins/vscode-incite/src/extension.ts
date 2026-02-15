import * as vscode from "vscode";
import { InCiteClient } from "./api-client";
import { CitationWatcher } from "./citation-watcher";
import { extractContext } from "./context-extractor";
import { SidebarProvider } from "./sidebar-provider";
import { StatusBar } from "./status-bar";
import type { InCiteSettings, Recommendation } from "./types";
import { DEFAULT_SETTINGS } from "./types";

let client: InCiteClient;
let statusBar: StatusBar;
let watcher: CitationWatcher | null = null;
let sidebarProvider: SidebarProvider;

/** Read extension settings from VS Code configuration. */
function getSettings(): InCiteSettings {
	const config = vscode.workspace.getConfiguration("incite");
	return {
		apiUrl: config.get<string>("apiUrl", DEFAULT_SETTINGS.apiUrl),
		k: config.get<number>("k", DEFAULT_SETTINGS.k),
		authorBoost: config.get<number>("authorBoost", DEFAULT_SETTINGS.authorBoost),
		contextSentences: config.get<number>("contextSentences", DEFAULT_SETTINGS.contextSentences),
		insertFormat: config.get<string>("insertFormat", DEFAULT_SETTINGS.insertFormat),
		autoDetectEnabled: config.get<boolean>("autoDetectEnabled", DEFAULT_SETTINGS.autoDetectEnabled),
		debounceMs: config.get<number>("debounceMs", DEFAULT_SETTINGS.debounceMs),
		showParagraphs: config.get<boolean>("showParagraphs", DEFAULT_SETTINGS.showParagraphs),
		citationPatterns: config.get<string[]>("citationPatterns", DEFAULT_SETTINGS.citationPatterns),
	};
}

/** Extract context at cursor and fetch recommendations. */
async function recommendAtCursor(): Promise<void> {
	const editor = vscode.window.activeTextEditor;
	if (!editor) {
		vscode.window.showWarningMessage("inCite: No active editor.");
		return;
	}

	const settings = getSettings();

	// Use selection if text is selected, otherwise extract context
	const selection = editor.document.getText(editor.selection);
	let queryText: string;

	if (selection && selection.trim().length > 0) {
		queryText = selection.trim();
	} else {
		const fullText = editor.document.getText();
		const cursorOffset = editor.document.offsetAt(editor.selection.active);

		const context = extractContext(
			fullText,
			cursorOffset,
			settings.contextSentences
		);
		queryText = context.text;
	}

	if (!queryText.trim()) {
		sidebarProvider.setError("No text around cursor to use as context.");
		return;
	}

	// Reveal the sidebar
	await vscode.commands.executeCommand("incite.sidebar.focus");

	sidebarProvider.setLoading();

	try {
		const response = await client.recommend(
			queryText,
			settings.k,
			settings.authorBoost
		);
		sidebarProvider.setResults(response.recommendations);
	} catch (err) {
		const message = err instanceof Error ? err.message : "Failed to connect";
		sidebarProvider.setError(
			`Could not reach inCite server at ${settings.apiUrl}. ` +
			`Is 'incite serve' running? (${message})`
		);
		vscode.window.showWarningMessage(
			"inCite: Server not reachable. Run 'incite serve'."
		);
	}
}

/** Insert a formatted citation at the current cursor position. */
function insertCitation(rec: Recommendation): void {
	const editor = vscode.window.activeTextEditor;
	if (!editor) {
		vscode.window.showWarningMessage(
			"inCite: No active editor to insert citation into."
		);
		return;
	}

	const settings = getSettings();

	// Extract last name of first author
	let firstAuthor = "Unknown";
	if (rec.authors && rec.authors.length > 0) {
		const parts = rec.authors[0].trim().split(/\s+/);
		firstAuthor = parts[parts.length - 1];
		if (rec.authors.length > 1) {
			firstAuthor += " et al.";
		}
	}

	const citation = settings.insertFormat
		.replace("${bibtex_key}", rec.bibtex_key || rec.paper_id)
		.replace("${paper_id}", rec.paper_id)
		.replace("${first_author}", firstAuthor)
		.replace("${year}", String(rec.year ?? "n.d."))
		.replace("${title}", rec.title);

	editor.edit((editBuilder) => {
		editBuilder.insert(editor.selection.active, citation);
	}).then((success) => {
		if (success) {
			vscode.window.showInformationMessage(`Inserted: ${citation}`);
		}
	});
}

/** Start the citation pattern watcher. */
function startWatcher(context: vscode.ExtensionContext): void {
	if (watcher) return;
	const settings = getSettings();
	watcher = new CitationWatcher(settings, () => recommendAtCursor());
	context.subscriptions.push(watcher.start());
}

/** Stop the citation pattern watcher. */
function stopWatcher(): void {
	if (watcher) {
		watcher.stop();
		watcher = null;
	}
}

export function activate(context: vscode.ExtensionContext): void {
	const settings = getSettings();

	// Initialize API client
	client = new InCiteClient(settings.apiUrl);

	// Initialize status bar
	statusBar = new StatusBar(client);
	statusBar.startPolling();
	context.subscriptions.push({ dispose: () => statusBar.dispose() });

	// Initialize sidebar
	sidebarProvider = new SidebarProvider(
		context.extensionUri,
		settings,
		insertCitation
	);
	context.subscriptions.push(
		vscode.window.registerWebviewViewProvider(
			SidebarProvider.viewType,
			sidebarProvider
		)
	);

	// Register commands
	context.subscriptions.push(
		vscode.commands.registerCommand(
			"incite.recommendAtCursor",
			recommendAtCursor
		)
	);

	context.subscriptions.push(
		vscode.commands.registerCommand("incite.openSidebar", () => {
			vscode.commands.executeCommand("incite.sidebar.focus");
		})
	);

	context.subscriptions.push(
		vscode.commands.registerCommand("incite.toggleAutoDetect", () => {
			const config = vscode.workspace.getConfiguration("incite");
			const current = config.get<boolean>("autoDetectEnabled", true);
			config.update("autoDetectEnabled", !current, vscode.ConfigurationTarget.Global);
			vscode.window.showInformationMessage(
				`inCite auto-detection ${!current ? "enabled" : "disabled"}.`
			);
		})
	);

	// Start watcher if auto-detect is enabled
	if (settings.autoDetectEnabled) {
		startWatcher(context);
	}

	// React to configuration changes
	context.subscriptions.push(
		vscode.workspace.onDidChangeConfiguration((e) => {
			if (!e.affectsConfiguration("incite")) return;

			const newSettings = getSettings();

			// Update API client URL
			client.setBaseUrl(newSettings.apiUrl);
			statusBar.updateClient(client);

			// Update sidebar settings
			sidebarProvider.updateSettings(newSettings);

			// Toggle watcher
			if (newSettings.autoDetectEnabled) {
				if (watcher) {
					watcher.updateSettings(newSettings);
				} else {
					startWatcher(context);
				}
			} else {
				stopWatcher();
			}
		})
	);
}

export function deactivate(): void {
	stopWatcher();
}
