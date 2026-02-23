/**
 * HTTP endpoints registered on Zotero's built-in connector server (port 23119).
 * Serves the browser panel and exposes Zotero settings to it.
 */
import { loadClientConfig, setPref } from "./prefs";
import { findPython, checkIncite, installIncite, getInstallState, startServer, stopManagedServer, processLibrary, getProcessStatus } from "./system-utils";
import { uploadToCloud, getUploadState } from "./cloud-upload";
import { syncFromCloud, getSyncState } from "./cloud-sync";

/** Panel HTML loaded at startup. */
let panelHtml = "";

/** JSON response helper. */
function jsonResponse(data: unknown): [number, string, string] {
	return [200, "application/json", JSON.stringify(data)];
}

function errorResponse(status: number, message: string): [number, string, string] {
	return [status, "application/json", JSON.stringify({ error: message })];
}

const ENDPOINT_PATHS = [
	"/incite/panel",
	"/incite/settings",
	"/incite/system/python",
	"/incite/system/incite",
	"/incite/system/install",
	"/incite/system/install/status",
	"/incite/system/server/start",
	"/incite/system/server/stop",
	"/incite/system/process",
	"/incite/system/process/status",
	"/incite/cloud/status",
	"/incite/cloud/sync",
	"/incite/cloud/sync/status",
	"/incite/cloud/upload",
	"/incite/cloud/upload/status",
] as const;

/** Register all inCite HTTP endpoints on Zotero's connector server. */
export function registerServerEndpoints(): void {
	// Pre-load panel HTML synchronously at startup
	try {
		const xhr = new XMLHttpRequest();
		xhr.open("GET", rootURI + "content/panel/panel.html", false);
		xhr.send();
		if (xhr.status === 200 || xhr.status === 0) {
			panelHtml = xhr.responseText;
		} else {
			Zotero.debug(`inCite: failed to load panel HTML (status ${xhr.status})`);
		}
	} catch (e) {
		Zotero.debug(`inCite: error loading panel HTML: ${e}`);
	}

	// GET /incite/panel — serve the self-contained HTML panel
	const PanelEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	PanelEndpoint.prototype = {
		supportedMethods: ["GET"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		init(_options) {
			if (!panelHtml) {
				return errorResponse(500, "Panel HTML not loaded");
			}
			return [200, "text/html", panelHtml];
		},
	};
	Zotero.Server.Endpoints["/incite/panel"] = PanelEndpoint;
	Zotero.debug(`inCite: registered /incite/panel (HTML ${panelHtml.length} bytes)`);

	// GET/PUT /incite/settings — read or write plugin settings
	const SettingsEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	SettingsEndpoint.prototype = {
		supportedMethods: ["GET", "POST"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		init(options) {
			if (options.method === "GET" || options.method === "get") {
				return jsonResponse(loadClientConfig());
			}

			// PUT — write settings
			try {
				const data = (typeof options.data === "string" ? JSON.parse(options.data) : options.data) as Record<string, unknown>;
				const allowedKeys = ["apiMode", "cloudUrl", "localUrl", "apiToken"];
				for (const key of allowedKeys) {
					if (key in data) {
						setPref(key, data[key] as string | number | boolean);
					}
				}
				return jsonResponse(loadClientConfig());
			} catch (e) {
				return errorResponse(400, `Invalid settings data: ${e}`);
			}
		},
	};
	Zotero.Server.Endpoints["/incite/settings"] = SettingsEndpoint;

	// GET /incite/system/python — detect Python installation
	const PythonEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	PythonEndpoint.prototype = {
		supportedMethods: ["GET"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		async init() {
			try {
				return jsonResponse(await findPython());
			} catch (e) {
				return errorResponse(500, String(e));
			}
		},
	};
	Zotero.Server.Endpoints["/incite/system/python"] = PythonEndpoint;

	// GET /incite/system/incite — check incite installation
	const InciteCheckEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	InciteCheckEndpoint.prototype = {
		supportedMethods: ["GET"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		async init() {
			try {
				const python = await findPython();
				if (!python.found || !python.path) {
					return jsonResponse({ installed: false, pythonFound: false });
				}
				const incite = await checkIncite(python.path);
				return jsonResponse({ ...incite, pythonFound: true, pythonPath: python.path });
			} catch (e) {
				return errorResponse(500, String(e));
			}
		},
	};
	Zotero.Server.Endpoints["/incite/system/incite"] = InciteCheckEndpoint;

	// POST /incite/system/install — start pip install
	const InstallEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	InstallEndpoint.prototype = {
		supportedMethods: ["POST"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		async init() {
			try {
				// Fire and forget — poll /incite/system/install/status
				// installIncite manages uv, Python, and venv itself
				installIncite();
				return jsonResponse({ status: "started" });
			} catch (e) {
				return errorResponse(500, String(e));
			}
		},
	};
	Zotero.Server.Endpoints["/incite/system/install"] = InstallEndpoint;

	// GET /incite/system/install/status — poll install progress
	const InstallStatusEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	InstallStatusEndpoint.prototype = {
		supportedMethods: ["GET"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		init() {
			return jsonResponse(getInstallState());
		},
	};
	Zotero.Server.Endpoints["/incite/system/install/status"] = InstallStatusEndpoint;

	// POST /incite/system/server/start — start incite serve
	const ServerStartEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	ServerStartEndpoint.prototype = {
		supportedMethods: ["POST"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		async init(options) {
			try {
				const python = await findPython();
				if (!python.found || !python.path) {
					return errorResponse(400, "Python not found");
				}
				const data = typeof options.data === "string" ? JSON.parse(options.data) : (options.data ?? {});
				const result = await startServer(python.path, (data as Record<string, unknown>).embedder as string | undefined);
				return jsonResponse(result);
			} catch (e) {
				return errorResponse(500, String(e));
			}
		},
	};
	Zotero.Server.Endpoints["/incite/system/server/start"] = ServerStartEndpoint;

	// POST /incite/system/server/stop — stop managed server
	const ServerStopEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	ServerStopEndpoint.prototype = {
		supportedMethods: ["POST"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		async init() {
			try {
				await stopManagedServer();
				return jsonResponse({ stopped: true });
			} catch (e) {
				return errorResponse(500, String(e));
			}
		},
	};
	Zotero.Server.Endpoints["/incite/system/server/stop"] = ServerStopEndpoint;

	// POST /incite/system/process — start local library processing
	const ProcessEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	ProcessEndpoint.prototype = {
		supportedMethods: ["POST"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		init() {
			try {
				processLibrary();
				return jsonResponse({ status: "started" });
			} catch (e) {
				return errorResponse(500, String(e));
			}
		},
	};
	Zotero.Server.Endpoints["/incite/system/process"] = ProcessEndpoint;

	// GET /incite/system/process/status — poll processing progress
	const ProcessStatusEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	ProcessStatusEndpoint.prototype = {
		supportedMethods: ["GET"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		async init() {
			try {
				return jsonResponse(await getProcessStatus());
			} catch (e) {
				return errorResponse(500, String(e));
			}
		},
	};
	Zotero.Server.Endpoints["/incite/system/process/status"] = ProcessStatusEndpoint;

	// GET /incite/cloud/status — proxy cloud library status (avoids CORS from panel)
	const CloudStatusEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	CloudStatusEndpoint.prototype = {
		supportedMethods: ["GET"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		async init() {
			try {
				const config = loadClientConfig();
				const serverUrl = (config.cloudUrl || "").replace(/\/+$/, "");
				const apiToken = config.apiToken || "";
				if (!serverUrl || !apiToken) {
					return jsonResponse({ library_status: "no_token", num_papers: 0, num_chunks: 0, grobid_fulltext_papers: 0, grobid_fulltext_chunks: 0, abstract_only_papers: 0 });
				}
				const resp = await Zotero.HTTP.request("GET", `${serverUrl}/api/v1/upload-library/status`, {
					headers: { Authorization: `Bearer ${apiToken}`, Accept: "application/json" },
					responseType: "text",
					timeout: 15000,
				});
				if (resp.status >= 200 && resp.status < 300) {
					return [200, "application/json", resp.responseText];
				}
				return errorResponse(resp.status, resp.responseText);
			} catch (e) {
				return errorResponse(502, `Cloud request failed: ${e}`);
			}
		},
	};
	Zotero.Server.Endpoints["/incite/cloud/status"] = CloudStatusEndpoint;

	// POST /incite/cloud/sync — sync papers from cloud into Zotero
	const CloudSyncEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	CloudSyncEndpoint.prototype = {
		supportedMethods: ["POST"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		init() {
			try {
				const config = loadClientConfig();
				const serverUrl = (config.cloudUrl || "").replace(/\/+$/, "");
				const apiToken = config.apiToken || "";
				if (!serverUrl || !apiToken) {
					return errorResponse(400, "Missing cloud URL or API token");
				}
				// Fire and forget — poll /incite/cloud/sync/status
				syncFromCloud(serverUrl, apiToken);
				return jsonResponse({ status: "started" });
			} catch (e) {
				return errorResponse(500, String(e));
			}
		},
	};
	Zotero.Server.Endpoints["/incite/cloud/sync"] = CloudSyncEndpoint;

	// GET /incite/cloud/sync/status — poll sync progress
	const CloudSyncStatusEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	CloudSyncStatusEndpoint.prototype = {
		supportedMethods: ["GET"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		init() {
			return jsonResponse(getSyncState());
		},
	};
	Zotero.Server.Endpoints["/incite/cloud/sync/status"] = CloudSyncStatusEndpoint;

	// POST /incite/cloud/upload — start cloud upload
	const CloudUploadEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	CloudUploadEndpoint.prototype = {
		supportedMethods: ["POST"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		init(options) {
			try {
				const config = loadClientConfig();
				const data = typeof options.data === "string" ? JSON.parse(options.data) : (options.data ?? {});
				const serverUrl = (data as Record<string, unknown>).serverUrl as string || config.cloudUrl || "";
				const apiToken = (data as Record<string, unknown>).apiToken as string || config.apiToken || "";
				if (!serverUrl || !apiToken) {
					return errorResponse(400, "Missing serverUrl or apiToken");
				}
				uploadToCloud(serverUrl, apiToken);
				return jsonResponse({ status: "started" });
			} catch (e) {
				return errorResponse(500, String(e));
			}
		},
	};
	Zotero.Server.Endpoints["/incite/cloud/upload"] = CloudUploadEndpoint;

	// GET /incite/cloud/upload/status — poll upload progress
	const CloudUploadStatusEndpoint = function () {} as unknown as Zotero.Server.EndpointConstructor;
	CloudUploadStatusEndpoint.prototype = {
		supportedMethods: ["GET"],
		supportedDataTypes: ["application/json"],
		permitBookmarklet: true,
		init() {
			return jsonResponse(getUploadState());
		},
	};
	Zotero.Server.Endpoints["/incite/cloud/upload/status"] = CloudUploadStatusEndpoint;

	Zotero.debug("inCite: all server endpoints registered");
}

/** Remove all inCite HTTP endpoints from Zotero's connector server. */
export function unregisterServerEndpoints(): void {
	for (const path of ENDPOINT_PATHS) {
		delete Zotero.Server.Endpoints[path];
	}
}
