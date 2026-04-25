/**
 * Overlay message types — communication between service worker and
 * the content script overlay (rail + command palette popup).
 */

/** Toggle the command palette popup in the overlay content script. */
export interface ToggleCommandPaletteMessage {
  type: "TOGGLE_COMMAND_PALETTE";
}

/** Request the service worker to open the Chrome side panel (Mode D). */
export interface OpenSidePanelMessage {
  type: "OPEN_SIDE_PANEL";
}

/** All overlay-specific messages. */
export type OverlayMessage = ToggleCommandPaletteMessage | OpenSidePanelMessage;
