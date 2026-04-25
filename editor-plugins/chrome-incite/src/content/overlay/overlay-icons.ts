/**
 * 16×16 stroke-only SVG icons for the overlay rail and popup.
 * All icons use `currentColor` for stroke so they inherit text color.
 */

/** Sparkle icon — trigger recommendations (Mode B). */
export const ICON_RECOMMEND = `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M8 1.5v3M8 11.5v3M14.5 8h-3M4.5 8h-3M12.1 3.9l-2.1 2.1M6 10l-2.1 2.1M12.1 12.1L10 10M6 6L3.9 3.9" stroke="currentColor" stroke-width="1.25" stroke-linecap="round"/>
</svg>`;

/** Chevron right — hide the rail (re-enable from extension options). */
export const ICON_HIDE = `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M6 4l4 4-4 4" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round"/>
</svg>`;

/** Sidebar expand — open full sidebar (Mode D). */
export const ICON_EXPAND = `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="2" y="2.5" width="12" height="11" rx="1.5" stroke="currentColor" stroke-width="1.25"/>
  <path d="M10 2.5v11" stroke="currentColor" stroke-width="1.25"/>
  <path d="M5.5 7L3.5 8.5 5.5 10" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round"/>
</svg>`;

/** X mark — close popup. */
export const ICON_CLOSE = `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M4.5 4.5l7 7M11.5 4.5l-7 7" stroke="currentColor" stroke-width="1.25" stroke-linecap="round"/>
</svg>`;

/** Gear — settings. */
export const ICON_SETTINGS = `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="8" cy="8" r="2" stroke="currentColor" stroke-width="1.25"/>
  <path d="M8 1.5v2M8 12.5v2M2.8 4.3l1.7 1M11.5 10.7l1.7 1M1.5 8h2M12.5 8h2M2.8 11.7l1.7-1M11.5 5.3l1.7-1" stroke="currentColor" stroke-width="1.25" stroke-linecap="round"/>
</svg>`;

/** InCite logo mark for popup header — simple "i" in a circle. */
export const ICON_LOGO = `<svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="9" cy="9" r="7.5" stroke="currentColor" stroke-width="1.25"/>
  <path d="M9 5.5v0M9 7.5v5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
</svg>`;
