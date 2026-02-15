import type { InCiteSettings } from "./types";

/**
 * Core citation watcher logic shared across editors.
 *
 * Handles debouncing and pattern matching. Each editor plugin provides
 * a thin adapter that hooks into its event system and calls checkLine().
 */
export class CitationWatcherCore {
	settings: InCiteSettings;
	private onTrigger: () => void;
	private debounceTimer: ReturnType<typeof setTimeout> | null = null;

	constructor(settings: InCiteSettings, onTrigger: () => void) {
		this.settings = settings;
		this.onTrigger = onTrigger;
	}

	/** Update settings reference (e.g., when debounce or patterns change). */
	updateSettings(settings: InCiteSettings): void {
		this.settings = settings;
	}

	/** Check a line of text against citation patterns and debounce-trigger. */
	checkLine(line: string): void {
		const hasMatch = this.settings.citationPatterns.some((pattern) => {
			try {
				return new RegExp(pattern).test(line);
			} catch {
				return false;
			}
		});

		if (!hasMatch) return;

		if (this.debounceTimer) {
			clearTimeout(this.debounceTimer);
		}

		this.debounceTimer = setTimeout(() => {
			this.debounceTimer = null;
			this.onTrigger();
		}, this.settings.debounceMs);
	}

	/** Clean up timers. */
	dispose(): void {
		if (this.debounceTimer) {
			clearTimeout(this.debounceTimer);
			this.debounceTimer = null;
		}
	}
}
