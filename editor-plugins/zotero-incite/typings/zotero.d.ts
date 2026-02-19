/** Minimal type declarations for Zotero 7 plugin APIs. */

declare namespace Zotero {
	function getMainWindows(): Window[];
	function debug(message: string, level?: number): void;
	function log(message: string): void;
	function getString(name: string): string;
	function launchURL(url: string): void;

	namespace HTTP {
		function request(
			method: string,
			url: string,
			options?: {
				headers?: Record<string, string>;
				body?: string;
				responseType?: string;
				timeout?: number;
			}
		): Promise<{
			status: number;
			responseText: string;
			getResponseHeader(name: string): string | null;
		}>;
	}

	namespace Prefs {
		function get(pref: string, global?: boolean): string | number | boolean | undefined;
		function set(pref: string, value: string | number | boolean, global?: boolean): void;
		function registerObserver(handler: (pref: string) => void): number;
		function unregisterObserver(id: number): void;
	}

	namespace ItemPaneManager {
		interface SectionOptions {
			paneID: string;
			pluginID: string;
			header: {
				l10nID?: string;
				label?: string;
				icon: string;
			};
			sidenav: {
				l10nID?: string;
				label?: string;
				icon: string;
			};
			onRender: (args: {
				body: HTMLElement;
				item: any;
				editable: boolean;
				tabType: string;
			}) => void;
			onAsyncRender?: (args: {
				body: HTMLElement;
				item: any;
				editable: boolean;
				tabType: string;
			}) => Promise<void>;
			onItemChange?: (args: {
				body: HTMLElement;
				item: any;
				tabType: string;
				setEnabled: (enabled: boolean) => void;
			}) => void;
		}

		function registerSection(options: SectionOptions): void;
		function unregisterSection(paneID: string): void;
	}

	namespace PreferencePanes {
		interface PaneOptions {
			pluginID: string;
			src: string;
			label?: string;
			l10nID?: string;
			image?: string;
		}

		function register(options: PaneOptions): void;
	}

	/** Zotero item (simplified). */
	interface Item {
		itemType: string;
		getField(field: string): string;
		getCreators(): Array<{
			firstName?: string;
			lastName?: string;
			creatorType: string;
		}>;
		getTags(): Array<{ tag: string }>;
		getNote?(): string;
		isRegularItem(): boolean;
		isNote(): boolean;
		isAttachment(): boolean;
	}
}

/** Mozilla XPCOM Components global. */
declare var Components: {
	classes: Record<string, any>;
	interfaces: Record<string, any>;
};

declare namespace Services {
	namespace scriptloader {
		function loadSubScript(url: string, scope?: object): void;
	}
}

/** Root URI set by bootstrap.js startup(). */
declare var rootURI: string;

/** Global InciteZotero set by the IIFE bundle. */
declare var InciteZotero: {
	hooks: {
		onStartup(): void;
		onMainWindowLoad(win: Window): void;
		onMainWindowUnload(win: Window): void;
		onShutdown(): void;
	};
};
