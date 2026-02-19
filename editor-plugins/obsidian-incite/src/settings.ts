import { App, Notice, PluginSettingTab, Setting } from "obsidian";
import type InCitePlugin from "./main";

export class InCiteSettingTab extends PluginSettingTab {
	plugin: InCitePlugin;

	constructor(app: App, plugin: InCitePlugin) {
		super(app, plugin);
		this.plugin = plugin;
	}

	display(): void {
		const { containerEl } = this;
		containerEl.empty();

		// --- Server ---
		containerEl.createEl("h3", { text: "Server" });

		new Setting(containerEl)
			.setName("API mode")
			.setDesc("Connect to the inCite cloud service or a local server")
			.addDropdown((dropdown) =>
				dropdown
					.addOption("cloud", "Cloud (inciteref.com)")
					.addOption("local", "Local (incite serve)")
					.setValue(this.plugin.settings.apiMode)
					.onChange(async (value) => {
						this.plugin.settings.apiMode = value as "cloud" | "local";
						await this.plugin.saveSettings();
						// Re-render to show/hide mode-specific fields
						this.display();
					})
			);

		if (this.plugin.settings.apiMode === "cloud") {
			new Setting(containerEl)
				.setName("API token")
				.setDesc("Your inCite API token (from inciteref.com account settings)")
				.addText((text) =>
					text
						.setPlaceholder("Enter your API token")
						.setValue(this.plugin.settings.apiToken)
						.onChange(async (value) => {
							this.plugin.settings.apiToken = value;
							await this.plugin.saveSettings();
						})
				);

			new Setting(containerEl)
				.setName("Cloud URL")
				.setDesc("inCite cloud server URL")
				.addText((text) =>
					text
						.setPlaceholder("https://inciteref.com")
						.setValue(this.plugin.settings.cloudUrl)
						.onChange(async (value) => {
							this.plugin.settings.cloudUrl = value;
							await this.plugin.saveSettings();
						})
				);
		} else {
			new Setting(containerEl)
				.setName("Local URL")
				.setDesc("URL of the local inCite server (run 'incite serve')")
				.addText((text) =>
					text
						.setPlaceholder("http://127.0.0.1:8230")
						.setValue(this.plugin.settings.localUrl)
						.onChange(async (value) => {
							this.plugin.settings.localUrl = value;
							await this.plugin.saveSettings();
						})
				);
		}

		new Setting(containerEl)
			.setName("Test connection")
			.setDesc("Check if the inCite server is reachable")
			.addButton((btn) =>
				btn.setButtonText("Test").onClick(async () => {
					try {
						const health = await this.plugin.client.health();
						if (health.ready) {
							new Notice(
								`Connected! ${health.corpus_size} papers, mode: ${health.mode || "paper"}`
							);
						} else {
							new Notice("Server is loading, try again in a moment.");
						}
					} catch {
						if (this.plugin.settings.apiMode === "cloud") {
							new Notice(
								"Could not connect. Check your API token and server URL."
							);
						} else {
							new Notice(
								"Could not connect. Is 'incite serve' running?"
							);
						}
					}
				})
			);

		// --- Retrieval ---
		containerEl.createEl("h3", { text: "Retrieval" });

		new Setting(containerEl)
			.setName("Number of results")
			.setDesc("How many recommendations to show")
			.addSlider((slider) =>
				slider
					.setLimits(1, 50, 1)
					.setValue(this.plugin.settings.k)
					.setDynamicTooltip()
					.onChange(async (value) => {
						this.plugin.settings.k = value;
						await this.plugin.saveSettings();
					})
			);

		new Setting(containerEl)
			.setName("Author boost")
			.setDesc(
				"Boost papers whose authors appear in context (1.0 = no boost, 1.2 = 20% boost)"
			)
			.addText((text) =>
				text
					.setPlaceholder("1.0")
					.setValue(String(this.plugin.settings.authorBoost))
					.onChange(async (value) => {
						const parsed = parseFloat(value);
						if (!isNaN(parsed) && parsed >= 0 && parsed <= 5) {
							this.plugin.settings.authorBoost = parsed;
							await this.plugin.saveSettings();
						}
					})
			);

		new Setting(containerEl)
			.setName("Context sentences")
			.setDesc("Total sentences to extract around cursor (split before/after)")
			.addSlider((slider) =>
				slider
					.setLimits(2, 20, 1)
					.setValue(this.plugin.settings.contextSentences)
					.setDynamicTooltip()
					.onChange(async (value) => {
						this.plugin.settings.contextSentences = value;
						await this.plugin.saveSettings();
					})
			);

		// --- Citation format ---
		containerEl.createEl("h3", { text: "Citation format" });

		new Setting(containerEl)
			.setName("Insert format")
			.setDesc(
				"Placeholders: {first_author}, {year}, {paper_id}, {bibtex_key}, {title}"
			)
			.addText((text) =>
				text
					.setPlaceholder("[({first_author}, {year})](zotero://.../{paper_id})")
					.setValue(this.plugin.settings.insertFormat)
					.onChange(async (value) => {
						this.plugin.settings.insertFormat = value;
						await this.plugin.saveSettings();
					})
			);

		// --- Auto-detection ---
		containerEl.createEl("h3", { text: "Auto-detection" });

		new Setting(containerEl)
			.setName("Enable auto-detection")
			.setDesc(
				"Automatically trigger recommendations when a citation marker is detected"
			)
			.addToggle((toggle) =>
				toggle
					.setValue(this.plugin.settings.autoDetectEnabled)
					.onChange(async (value) => {
						this.plugin.settings.autoDetectEnabled = value;
						await this.plugin.saveSettings();
					})
			);

		new Setting(containerEl)
			.setName("Debounce delay (ms)")
			.setDesc("Wait this long after typing before triggering (default: 500)")
			.addText((text) =>
				text
					.setPlaceholder("500")
					.setValue(String(this.plugin.settings.debounceMs))
					.onChange(async (value) => {
						const parsed = parseInt(value);
						if (!isNaN(parsed) && parsed >= 100 && parsed <= 5000) {
							this.plugin.settings.debounceMs = parsed;
							await this.plugin.saveSettings();
						}
					})
			);

		// --- Display ---
		containerEl.createEl("h3", { text: "Display" });

		new Setting(containerEl)
			.setName("Show matched paragraphs")
			.setDesc(
				"Display paragraph evidence when the server runs in paragraph mode"
			)
			.addToggle((toggle) =>
				toggle
					.setValue(this.plugin.settings.showParagraphs)
					.onChange(async (value) => {
						this.plugin.settings.showParagraphs = value;
						await this.plugin.saveSettings();
					})
			);
	}
}
