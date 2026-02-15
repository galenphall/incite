import type { ChromeExtensionSettings } from "./types";

export const DEFAULT_SETTINGS: ChromeExtensionSettings = {
  apiMode: "cloud",
  cloudUrl: "https://inciteref.com",
  localUrl: "http://localhost:8230",
  apiToken: "",
  k: 10,
  authorBoost: 0,
  contextSentences: 6,
  googleDocsCitationFormat: "[@{bibtex_key}]",
  overleafCitationFormat: "\\cite{{{bibtex_key}}}",
  showParagraphs: true,
  showAbstracts: false,
};

export const STORAGE_KEY = "incite_settings";
