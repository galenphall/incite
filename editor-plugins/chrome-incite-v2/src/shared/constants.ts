import { DEFAULT_SETTINGS as SHARED_DEFAULTS } from "@incite/shared";
import type { ChromeExtensionSettings } from "./types";

export const DEFAULT_SETTINGS: ChromeExtensionSettings = {
  ...SHARED_DEFAULTS,
  citationStyle: "apa",
  googleDocsCitationFormat: "({first_author}, {year})",
  overleafCitationFormat: "\\cite{{{bibtex_key}}}",
  showAbstracts: false,
};

export const STORAGE_KEY = "incite_settings";
