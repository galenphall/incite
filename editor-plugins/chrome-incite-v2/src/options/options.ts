import type { ChromeExtensionSettings } from "../shared/types";
import { DEFAULT_SETTINGS } from "../shared/constants";
import { loadSettings, saveSettings } from "../shared/settings";

// Citation style presets
const CITATION_PRESETS: Record<string, string> = {
  apa: "({first_author}, {year})",
  narrative: "{first_author} ({year})",
  mla: "({first_author})",
  harvard: "({first_author} {year})",
  bibtex: "[@{bibtex_key}]",
  latex: "\\cite{{{bibtex_key}}}",
};

// --- DOM references ---
const citationStyle = document.getElementById("citationStyle") as HTMLSelectElement;
const customFormatField = document.getElementById("custom-format-field")!;
const customCitationFormat = document.getElementById("customCitationFormat") as HTMLInputElement;
const apiToken = document.getElementById("apiToken") as HTMLInputElement;
const kInput = document.getElementById("k") as HTMLInputElement;
const showParagraphs = document.getElementById("showParagraphs") as HTMLInputElement;
const showAbstracts = document.getElementById("showAbstracts") as HTMLInputElement;
const apiModeRadios = document.querySelectorAll<HTMLInputElement>('input[name="apiMode"]');
const localFields = document.getElementById("local-fields")!;
const localUrl = document.getElementById("localUrl") as HTMLInputElement;
const contextSentences = document.getElementById("contextSentences") as HTMLInputElement;
const overleafFmt = document.getElementById("overleafCitationFormat") as HTMLInputElement;
const btnTest = document.getElementById("btn-test")!;
const testResult = document.getElementById("test-result")!;
const btnSave = document.getElementById("btn-save")!;
const saveStatus = document.getElementById("save-status")!;

// --- Load settings on page open ---
loadSettings().then(populateForm);

// --- Citation style toggle ---
citationStyle.addEventListener("change", () => {
  customFormatField.classList.toggle("hidden", citationStyle.value !== "custom");
});

// --- API mode toggle ---
apiModeRadios.forEach((radio) => {
  radio.addEventListener("change", () => {
    localFields.classList.toggle("hidden", radio.value !== "local");
  });
});

// --- Test connection (saves first to use current form values) ---
btnTest.addEventListener("click", async () => {
  testResult.textContent = "Testing...";
  testResult.className = "test-result";

  // Save current form values first so the health check uses them
  const selectedMode = document.querySelector<HTMLInputElement>('input[name="apiMode"]:checked');
  const currentSettings: Partial<ChromeExtensionSettings> = {
    apiMode: (selectedMode?.value as "cloud" | "local") ?? "cloud",
    cloudUrl: DEFAULT_SETTINGS.cloudUrl,
    localUrl: localUrl.value.trim() || DEFAULT_SETTINGS.localUrl,
    apiToken: apiToken.value.trim(),
  };
  await saveSettings(currentSettings);

  try {
    const response = await chrome.runtime.sendMessage({ type: "CHECK_HEALTH" });
    if (response?.response) {
      testResult.textContent = `Connected -- ${response.response.corpus_size ?? "?"} papers (${response.response.mode ?? "unknown"} mode)`;
      testResult.className = "test-result success";
    } else {
      testResult.textContent = response?.error ?? "Connection failed";
      testResult.className = "test-result error";
    }
  } catch (err: unknown) {
    testResult.textContent = err instanceof Error ? err.message : "Connection failed";
    testResult.className = "test-result error";
  }
});

// --- Save ---
btnSave.addEventListener("click", async () => {
  const selectedMode = document.querySelector<HTMLInputElement>('input[name="apiMode"]:checked');

  // Determine Google Docs citation format from style picker
  let googleDocsFmtValue: string;
  if (citationStyle.value === "custom") {
    googleDocsFmtValue = customCitationFormat.value || DEFAULT_SETTINGS.googleDocsCitationFormat;
  } else {
    googleDocsFmtValue = CITATION_PRESETS[citationStyle.value] || DEFAULT_SETTINGS.googleDocsCitationFormat;
  }

  const settings: Partial<ChromeExtensionSettings> = {
    apiMode: (selectedMode?.value as "cloud" | "local") ?? "cloud",
    cloudUrl: DEFAULT_SETTINGS.cloudUrl,
    localUrl: localUrl.value.trim() || DEFAULT_SETTINGS.localUrl,
    apiToken: apiToken.value.trim(),
    k: parseInt(kInput.value) || DEFAULT_SETTINGS.k,
    contextSentences: parseInt(contextSentences.value) || DEFAULT_SETTINGS.contextSentences,
    citationStyle: citationStyle.value,
    googleDocsCitationFormat: googleDocsFmtValue,
    overleafCitationFormat: overleafFmt.value || DEFAULT_SETTINGS.overleafCitationFormat,
    showParagraphs: showParagraphs.checked,
    showAbstracts: showAbstracts.checked,
  };

  await saveSettings(settings);
  saveStatus.textContent = "Saved!";
  setTimeout(() => { saveStatus.textContent = ""; }, 2000);
});

// --- Populate form from saved settings ---
function populateForm(settings: ChromeExtensionSettings) {
  // Citation style
  citationStyle.value = settings.citationStyle || "apa";
  customFormatField.classList.toggle("hidden", citationStyle.value !== "custom");
  if (citationStyle.value === "custom") {
    customCitationFormat.value = settings.googleDocsCitationFormat;
  }

  // Connection
  apiToken.value = settings.apiToken;

  // Display
  kInput.value = String(settings.k);
  showParagraphs.checked = settings.showParagraphs;
  showAbstracts.checked = settings.showAbstracts;

  // Advanced
  apiModeRadios.forEach((radio) => {
    radio.checked = radio.value === settings.apiMode;
  });
  localFields.classList.toggle("hidden", settings.apiMode !== "local");
  localUrl.value = settings.localUrl;
  contextSentences.value = String(settings.contextSentences);
  overleafFmt.value = settings.overleafCitationFormat;
}
