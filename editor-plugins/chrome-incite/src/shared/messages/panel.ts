/**
 * Panel-to-service-worker message types.
 * These messages are sent from the side panel to request recommendations,
 * health checks, settings, and citation insertions.
 */

import type { Recommendation, RecommendResponse, HealthResponse } from "@incite/shared";
import type { ChromeExtensionSettings } from "../types";

export interface GetRecommendationsMessage {
  type: "GET_RECOMMENDATIONS";
  collectionId?: string | null;
}

export interface GetRecommendationsForTextMessage {
  type: "GET_RECOMMENDATIONS_FOR_TEXT";
  text: string;
  collectionId?: string | null;
}

export interface CheckHealthMessage {
  type: "CHECK_HEALTH";
}

export interface GetSettingsMessage {
  type: "GET_SETTINGS";
}

export interface SaveSettingsMessage {
  type: "SAVE_SETTINGS";
  settings: Partial<ChromeExtensionSettings>;
}

export interface InsertCitationRequestMessage {
  type: "INSERT_CITATION_REQUEST";
  recommendation: Recommendation;
}

export interface InsertMultiCitationRequestMessage {
  type: "INSERT_MULTI_CITATION_REQUEST";
  recommendations: Recommendation[];
}
