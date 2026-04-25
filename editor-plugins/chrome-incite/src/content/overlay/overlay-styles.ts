/**
 * CSS for the overlay shadow DOM — rail (Mode A) and command palette popup (Mode B).
 *
 * Design tokens mirror panel.css :root for visual consistency.
 * All styles are scoped to the shadow root via :host.
 */
export const OVERLAY_CSS = /* css */ `

/* ─── Design Tokens ─── */

:host {
  --ink:        #1a1a1e;
  --ink-soft:   #2a2a2f;
  --ink-muted:  #4a4a52;
  --graphite:   #6b6b74;
  --slate:      #8e8e97;
  --silver:     #b8b8bf;
  --fog:        #dcdce0;
  --paper:      #f0efe9;
  --paper-warm: #f5f4ed;
  --cream:      #faf9f3;

  --oxblood:       #8b2e2e;
  --oxblood-light: #a83c3c;

  --success: #3a6b4a;
  --warning: #8b6b2e;
  --error:   #8b2e2e;

  --font-display: 'Source Serif 4', Georgia, 'Times New Roman', serif;
  --font-body:    'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono:    'IBM Plex Mono', 'SF Mono', 'Consolas', monospace;

  /* Semantic */
  --bg: var(--cream);
  --fg: var(--ink);
  --fg-muted: var(--graphite);
  --border: var(--fog);
  --card-bg: #ffffff;
  --accent: var(--ink);
  --accent-hover: var(--ink-soft);
  --evidence-bg: var(--paper);
  --evidence-border: var(--silver);
  --badge-bg: var(--paper);
  --badge-fg: var(--ink-muted);
  --radius: 6px;
  --radius-sm: 3px;

  /* Dot colors */
  --dot-filled: var(--success);
  --dot-empty: var(--silver);

  font-family: var(--font-body);
  font-size: 12px;
  line-height: 1.45;
  color: var(--fg);
  -webkit-font-smoothing: antialiased;
}

@media (prefers-color-scheme: dark) {
  :host {
    --bg: var(--ink);
    --fg: var(--cream);
    --fg-muted: var(--slate);
    --border: #3a3a40;
    --card-bg: var(--ink-soft);
    --accent: var(--cream);
    --accent-hover: var(--silver);
    --evidence-bg: rgba(255, 255, 255, 0.06);
    --evidence-border: #4a4a52;
    --badge-bg: #3a3a40;
    --badge-fg: var(--silver);
    --dot-filled: #7abf8e;
    --dot-empty: #4a4a52;
  }
}

* { box-sizing: border-box; margin: 0; padding: 0; }

/* ─── Rail (Mode A) ─── */

.incite-rail {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  width: 40px;
  background: var(--paper);
  border-left: 1px solid var(--fog);
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px 0;
  gap: 2px;
  pointer-events: auto;
  z-index: 1;
}

@media (prefers-color-scheme: dark) {
  .incite-rail {
    background: var(--ink-soft);
    border-left-color: var(--border);
  }
}

.rail-btn {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  background: transparent;
  color: var(--fg-muted);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease;
  padding: 0;
}

.rail-btn:hover {
  background: var(--fog);
  color: var(--fg);
}

@media (prefers-color-scheme: dark) {
  .rail-btn:hover {
    background: rgba(255, 255, 255, 0.1);
  }
}

.rail-btn:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: -2px;
}

.rail-btn svg {
  width: 16px;
  height: 16px;
  flex-shrink: 0;
}

.rail-spacer {
  flex: 1;
}

.rail-divider {
  width: 20px;
  height: 1px;
  background: var(--border);
  margin: 4px 0;
}

/* ─── Popup (Mode B) ─── */

.incite-popup {
  position: fixed;
  top: 8px;
  right: 48px;
  width: 400px;
  max-height: min(520px, calc(100vh - 16px));
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12), 0 2px 8px rgba(0, 0, 0, 0.08);
  display: flex;
  flex-direction: column;
  pointer-events: auto;
  z-index: 2;
  animation: popup-in 0.15s ease-out;
}

@keyframes popup-in {
  from { opacity: 0; transform: translateY(-4px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}

.popup-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.popup-logo {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--fg);
  font-family: var(--font-display);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.01em;
}

.popup-logo svg {
  width: 18px;
  height: 18px;
  flex-shrink: 0;
}

.popup-close {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  background: transparent;
  color: var(--fg-muted);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease;
  padding: 0;
}

.popup-close:hover {
  background: var(--badge-bg);
  color: var(--fg);
}

.popup-close svg {
  width: 14px;
  height: 14px;
}

/* Status bar */

.popup-status {
  padding: 6px 12px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--fg-muted);
  letter-spacing: 0.02em;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

/* Results area */

.popup-results {
  flex: 1;
  overflow-y: auto;
  min-height: 0;
}

.popup-results::-webkit-scrollbar {
  width: 4px;
}

.popup-results::-webkit-scrollbar-track {
  background: transparent;
}

.popup-results::-webkit-scrollbar-thumb {
  background: var(--fog);
  border-radius: 2px;
}

/* ─── Result Row ─── */

.result-row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  transition: background 0.1s ease;
}

.result-row:last-child {
  border-bottom: none;
}

.result-row:hover {
  background: var(--paper-warm);
}

@media (prefers-color-scheme: dark) {
  .result-row:hover {
    background: rgba(255, 255, 255, 0.04);
  }
}

.result-row.selected {
  background: var(--paper);
}

@media (prefers-color-scheme: dark) {
  .result-row.selected {
    background: rgba(255, 255, 255, 0.08);
  }
}

/* Relevance dots */

.relevance-dots {
  display: flex;
  gap: 2px;
  padding-top: 4px;
  flex-shrink: 0;
}

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--dot-empty);
  transition: background 0.15s ease;
}

.dot.filled {
  background: var(--dot-filled);
}

/* Row content */

.row-content {
  flex: 1;
  min-width: 0;
}

.row-title {
  font-family: var(--font-display);
  font-size: 12px;
  font-weight: 400;
  line-height: 1.35;
  color: var(--fg);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.row-meta {
  font-size: 11px;
  color: var(--fg-muted);
  margin-top: 1px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Badges */

.row-badges {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
  flex-shrink: 0;
  padding-top: 2px;
}

.cited-badge {
  font-family: var(--font-mono);
  font-size: 9px;
  font-weight: 500;
  padding: 1px 5px;
  border-radius: var(--radius-sm);
  background: rgba(58, 107, 74, 0.15);
  color: var(--success);
  letter-spacing: 0.02em;
  white-space: nowrap;
}

.insert-hint {
  font-family: var(--font-mono);
  font-size: 9px;
  color: var(--fg-muted);
  opacity: 0;
  transition: opacity 0.1s ease;
}

.result-row.selected .insert-hint,
.result-row:hover .insert-hint {
  opacity: 1;
}

/* ─── Evidence Expansion ─── */

.row-evidence {
  padding: 4px 12px 8px 42px;
  border-bottom: 1px solid var(--border);
  animation: evidence-in 0.15s ease-out;
}

@keyframes evidence-in {
  from { opacity: 0; max-height: 0; }
  to   { opacity: 1; max-height: 300px; }
}

.evidence {
  font-size: 11px;
  color: var(--fg);
  border-left: 3px solid var(--evidence-border);
  background: var(--evidence-bg);
  padding: 6px 8px;
  margin: 4px 0;
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  line-height: 1.4;
}

.evidence-secondary {
  opacity: 0.75;
  font-size: 10px;
  margin-top: 4px;
}

.evidence-score {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 500;
  color: var(--ink-muted);
  margin-right: 4px;
}

@media (prefers-color-scheme: dark) {
  .evidence-score { color: var(--slate); }
}

/* ─── Loading Skeleton ─── */

.skeleton-row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
}

.skeleton-dots {
  display: flex;
  gap: 2px;
  padding-top: 4px;
}

.skeleton-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--fog);
}

.skeleton-lines {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.skeleton-line {
  height: 10px;
  border-radius: 3px;
  background: linear-gradient(90deg, var(--fog) 25%, var(--paper-warm) 50%, var(--fog) 75%);
  background-size: 400px 100%;
  animation: shimmer 1.5s ease-in-out infinite;
}

@media (prefers-color-scheme: dark) {
  .skeleton-dot { background: var(--border); }
  .skeleton-line {
    background: linear-gradient(90deg, var(--border) 25%, #4a4a52 50%, var(--border) 75%);
    background-size: 400px 100%;
  }
}

.skeleton-title { width: 75%; }
.skeleton-meta  { width: 45%; height: 8px; }

@keyframes shimmer {
  0%   { background-position: -200px 0; }
  100% { background-position: 200px 0; }
}

/* ─── Empty & Error States ─── */

.popup-empty,
.popup-error {
  padding: 32px 16px;
  text-align: center;
  color: var(--fg-muted);
  font-size: 12px;
  line-height: 1.5;
}

.popup-error {
  color: var(--error);
}

.popup-empty p + p {
  margin-top: 6px;
  font-size: 11px;
}

/* ─── Hint Bar ─── */

.popup-hints {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 6px 12px;
  border-top: 1px solid var(--border);
  font-size: 10px;
  color: var(--fg-muted);
  flex-shrink: 0;
}

.popup-hints kbd {
  display: inline-block;
  padding: 1px 4px;
  font-size: 9px;
  font-family: var(--font-mono);
  background: var(--badge-bg);
  border-radius: 3px;
  border: 1px solid var(--border);
  color: var(--fg-muted);
  line-height: 1.4;
}

/* ─── Toast ─── */

.overlay-toast {
  position: fixed;
  bottom: 16px;
  right: 56px;
  padding: 8px 16px;
  background: var(--ink);
  color: var(--cream);
  border-radius: var(--radius);
  font-family: var(--font-body);
  font-size: 12px;
  pointer-events: auto;
  z-index: 10;
  animation: toast-in 0.2s ease-out;
}

@keyframes toast-in {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

@media (prefers-color-scheme: dark) {
  .overlay-toast {
    background: var(--cream);
    color: var(--ink);
  }
}
`;
