/** Minimal type declarations for Zotero 7 plugin APIs. */

declare namespace Zotero {
	function getMainWindows(): Window[];
	function debug(message: string, level?: number): void;
	function log(message: string): void;
	function getString(name: string): string;
	function launchURL(url: string): void;

	/** Get the active ZoteroPane (item list, reader, etc.). */
	function getActiveZoteroPane(): {
		getSelectedItems(): Zotero.Item[];
	} | null;

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

	/**
	 * Zotero's built-in connector HTTP server (port 23119).
	 * Endpoints are registered as constructor functions with a prototype.
	 */
	namespace Server {
		/** Map of URL path → endpoint constructor. */
		const Endpoints: Record<string, EndpointConstructor>;

		interface EndpointConstructor {
			new (): EndpointInstance;
			prototype: EndpointInstance;
		}

		interface EndpointInstance {
			supportedMethods: string[];
			supportedDataTypes?: string[];
			permitBookmarklet?: boolean;
			init(options: {
				method: string;
				pathname: string;
				query?: Record<string, string>;
				headers?: Record<string, string>;
				data?: unknown;
			}): [number, string, string] | Promise<[number, string, string]>;
		}
	}

	namespace Utilities {
		namespace Internal {
			function exec(command: any, args?: string[]): Promise<{ exitCode: number }>;
		}
	}

	function getTempDirectory(): { path: string };

	const isWin: boolean;
	const isMac: boolean;
	const isLinux: boolean;

	namespace File {
		function getContentsAsync(path: string): Promise<string>;
		function putContentsAsync(path: string, data: string): Promise<void>;
	}

	namespace Items {
		function getAll(libraryID: number, onlyRegular?: boolean, includeDeleted?: boolean): Promise<Zotero.Item[]>;
		function getAsync(id: number): Promise<Zotero.Item>;
		function getByLibraryAndKey(libraryID: number, key: string): Zotero.Item | false;
	}

	namespace Collections {
		function getByLibrary(libraryID: number): Zotero.Collection[];
	}

	/** Zotero collection. */
	class Collection {
		id: number;
		libraryID: number;
		name: string;
		key: string;
		hasItem(itemID: number): boolean;
		addItem(itemID: number): void;
		saveTx(): Promise<void>;
	}

	namespace Libraries {
		const userLibraryID: number;
	}

	namespace ItemTypes {
		function getName(typeID: number): string;
	}

	namespace CreatorTypes {
		function getName(typeID: number): string;
	}

	/** Zotero item constructor — use `new Zotero.Item("journalArticle")`. */
	// eslint-disable-next-line @typescript-eslint/no-misused-new
	class Item {
		constructor(itemType?: string);
		id: number;
		key: string;
		libraryID: number;
		itemType: string;
		itemTypeID: number;
		getField(field: string): string;
		setField(field: string, value: string | number): void;
		getCreators(): Array<{
			firstName?: string;
			lastName?: string;
			creatorType: string;
		}>;
		setCreators(creators: Array<{ firstName: string; lastName: string; creatorType: string }>): void;
		getTags(): Array<{ tag: string }>;
		addTag(tag: string, type?: number): void;
		getNote?(): string;
		isRegularItem(): boolean;
		isNote(): boolean;
		isAnnotation(): boolean;
		isAttachment(): boolean;
		getAttachments(): number[];
		attachmentContentType?: string;
		getFilePathAsync(): Promise<string | false>;
		saveTx(): Promise<void>;
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
	namespace wm {
		function getMostRecentWindow(windowType: string): Window | null;
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

/** Mozilla IOUtils (Gecko global for file I/O). */
declare namespace IOUtils {
	function read(path: string): Promise<Uint8Array>;
}
