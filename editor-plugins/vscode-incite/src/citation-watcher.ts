import * as vscode from "vscode";
import { CitationWatcherCore } from "@incite/shared";
import type { InCiteSettings } from "./types";

/**
 * VS Code-specific citation watcher.
 *
 * Thin adapter around CitationWatcherCore that hooks into VS Code's
 * onDidChangeTextDocument event and provides the current line.
 */
export class CitationWatcher {
	private core: CitationWatcherCore;
	private disposable: vscode.Disposable | null = null;

	constructor(settings: InCiteSettings, onTrigger: () => void) {
		this.core = new CitationWatcherCore(settings, onTrigger);
	}

	/** Register the document change event listener. */
	start(): vscode.Disposable {
		this.disposable = vscode.workspace.onDidChangeTextDocument((e) => {
			const editor = vscode.window.activeTextEditor;
			if (!editor || editor.document !== e.document) return;

			const cursor = editor.selection.active;
			const line = e.document.lineAt(cursor.line).text;
			this.core.checkLine(line);
		});
		return this.disposable;
	}

	/** Unregister and clean up. */
	stop(): void {
		this.core.dispose();
		if (this.disposable) {
			this.disposable.dispose();
			this.disposable = null;
		}
	}

	/** Update settings reference. */
	updateSettings(settings: InCiteSettings): void {
		this.core.updateSettings(settings);
	}
}
