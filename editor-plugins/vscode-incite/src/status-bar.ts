import * as vscode from "vscode";
import { InCiteClient } from "./api-client";

/** Persistent status bar item showing server status and paper count. */
export class StatusBar {
	private item: vscode.StatusBarItem;
	private client: InCiteClient;
	private pollInterval: ReturnType<typeof setInterval> | null = null;

	constructor(client: InCiteClient) {
		this.client = client;
		this.item = vscode.window.createStatusBarItem(
			vscode.StatusBarAlignment.Right,
			100
		);
		this.item.command = "incite.openSidebar";
		this.item.tooltip = "inCite — click to open sidebar";
		this.setOffline();
		this.item.show();
	}

	/** Start polling the health endpoint every 30 seconds. */
	startPolling(): void {
		this.poll(); // immediate first check
		this.pollInterval = setInterval(() => this.poll(), 30_000);
	}

	/** Stop polling. */
	stopPolling(): void {
		if (this.pollInterval) {
			clearInterval(this.pollInterval);
			this.pollInterval = null;
		}
	}

	/** Update the client URL (when settings change). */
	updateClient(client: InCiteClient): void {
		this.client = client;
		this.poll();
	}

	/** Clean up the status bar item. */
	dispose(): void {
		this.stopPolling();
		this.item.dispose();
	}

	private async poll(): Promise<void> {
		try {
			const health = await this.client.health();
			if (health.ready && health.corpus_size !== undefined) {
				this.item.text = `$(book) ${health.corpus_size} papers`;
				this.item.backgroundColor = undefined;
			} else {
				this.item.text = "$(book) Loading...";
				this.item.backgroundColor = undefined;
			}
		} catch {
			this.setOffline();
		}
	}

	private setOffline(): void {
		this.item.text = "$(book) Offline";
		this.item.backgroundColor = new vscode.ThemeColor(
			"statusBarItem.warningBackground"
		);
	}
}
