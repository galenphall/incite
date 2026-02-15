import type { ChromeExtensionSettings } from "../shared/types";
import { DEFAULT_SETTINGS } from "../shared/constants";
import { loadSettings, saveSettings } from "../shared/settings";

// --- DOM references ---
const apiModeRadios = document.querySelectorAll<HTMLInputElement>('input[name="apiMode"]');
const cloudFields = document.getElementById("cloud-fields")!;
const localFields = document.getElementById("local-fields")!;
const cloudUrl = document.getElementById("cloudUrl") as HTMLInputElement;
const apiToken = document.getElementById("apiToken") as HTMLInputElement;
const localUrl = document.getElementById("localUrl") as HTMLInputElement;
const kInput = document.getElementById("k") as HTMLInputElement;
const contextSentences = document.getElementById("contextSentences") as HTMLInputElement;
const googleDocsFmt = document.getElementById("googleDocsCitationFormat") as HTMLInputElement;
const overleafFmt = document.getElementById("overleafCitationFormat") as HTMLInputElement;
const showParagraphs = document.getElementById("showParagraphs") as HTMLInputElement;
const showAbstracts = document.getElementById("showAbstracts") as HTMLInputElement;
const btnTest = document.getElementById("btn-test")!;
const testResult = document.getElementById("test-result")!;
const btnSave = document.getElementById("btn-save")!;
const saveStatus = document.getElementById("save-status")!;

// --- Load settings on page open ---
loadSettings().then(populateForm);

// --- API mode toggle ---
apiModeRadios.forEach((radio) => {
  radio.addEventListener("change", () => {
    const isCloud = radio.value === "cloud";
    cloudFields.classList.toggle("hidden", !isCloud);
    localFields.classList.toggle("hidden", isCloud);
  });
});

// --- Test connection ---
btnTest.addEventListener("click", async () => {
  testResult.textContent = "Testing...";
  testResult.className = "test-result";

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

  const settings: Partial<ChromeExtensionSettings> = {
    apiMode: (selectedMode?.value as "cloud" | "local") ?? "cloud",
    cloudUrl: cloudUrl.value.trim() || DEFAULT_SETTINGS.cloudUrl,
    localUrl: localUrl.value.trim() || DEFAULT_SETTINGS.localUrl,
    apiToken: apiToken.value.trim(),
    k: parseInt(kInput.value) || DEFAULT_SETTINGS.k,
    contextSentences: parseInt(contextSentences.value) || DEFAULT_SETTINGS.contextSentences,
    googleDocsCitationFormat: googleDocsFmt.value || DEFAULT_SETTINGS.googleDocsCitationFormat,
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
  // API mode
  apiModeRadios.forEach((radio) => {
    radio.checked = radio.value === settings.apiMode;
  });
  cloudFields.classList.toggle("hidden", settings.apiMode !== "cloud");
  localFields.classList.toggle("hidden", settings.apiMode !== "local");

  // URLs and token
  cloudUrl.value = settings.cloudUrl;
  apiToken.value = settings.apiToken;
  localUrl.value = settings.localUrl;

  // Recommendations
  kInput.value = String(settings.k);
  contextSentences.value = String(settings.contextSentences);

  // Citation formats
  googleDocsFmt.value = settings.googleDocsCitationFormat;
  overleafFmt.value = settings.overleafCitationFormat;

  // Display
  showParagraphs.checked = settings.showParagraphs;
  showAbstracts.checked = settings.showAbstracts;
}
