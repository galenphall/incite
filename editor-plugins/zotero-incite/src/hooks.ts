import { registerPreferencesPane } from "./prefs";
import { registerServerEndpoints, unregisterServerEndpoints } from "./server-endpoints";
import { stopManagedServer } from "./system-utils";
import { registerToolsMenu, unregisterToolsMenu } from "./text-query-dialog";

export const hooks = {
	onStartup(): void {
		registerPreferencesPane();
		registerServerEndpoints();
	},

	onMainWindowLoad(win: Window): void {
		registerToolsMenu(win);
	},

	onMainWindowUnload(win: Window): void {
		unregisterToolsMenu(win);
	},

	onShutdown(): void {
		stopManagedServer();
		unregisterServerEndpoints();
	},
};
