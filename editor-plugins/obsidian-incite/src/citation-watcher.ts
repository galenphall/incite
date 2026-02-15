import { App, Editor, EventRef, Plugin } from "obsidian";
import { CitationWatcherCore } from "@incite/shared";
import type { InCiteSettings } from "./types";

/**
 * Obsidian-specific citation watcher.
 *
 * Thin adapter around CitationWatcherCore that hooks into Obsidian's
 * editor-change event and provides the current line for pattern matching.
 */
export class CitationWatcher {
	private app: App;
	private core: CitationWatcherCore;
	private onTrigger: (editor: Editor) => void;
	private eventRef: EventRef | null = null;
	private lastEditor: Editor | null = null;

	constructor(
		app: App,
		settings: InCiteSettings,
		onTrigger: (editor: Editor) => void
	) {
		this.app = app;
		this.onTrigger = onTrigger;
		this.core = new CitationWatcherCore(settings, () => {
			if (this.lastEditor) {
				this.onTrigger(this.lastEditor);
			}
		});
	}

	/** Register the editor-change event listener. */
	start(plugin: Plugin): void {
		this.eventRef = this.app.workspace.on("editor-change", (editor: Editor) => {
			this.lastEditor = editor;
			const cursor = editor.getCursor();
			const line = editor.getLine(cursor.line);
			this.core.checkLine(line);
		});
		plugin.registerEvent(this.eventRef);
	}

	/** Unregister and clean up. */
	stop(): void {
		this.core.dispose();
		this.eventRef = null;
		this.lastEditor = null;
	}

	/** Update settings reference. */
	updateSettings(settings: InCiteSettings): void {
		this.core.updateSettings(settings);
	}
}
