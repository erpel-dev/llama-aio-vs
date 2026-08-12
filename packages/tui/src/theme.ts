/** Shared palette for the Llama AIO TUI. */
export const theme = {
  bg: "#12141a",
  panel: "#1a1d24",
  border: "#3a3f4b",
  borderFocus: "#3794ff",
  text: "#e6e6e6",
  muted: "#9da3af",
  accent: "#3794ff",
  ok: "#3fb950",
  warn: "#d29922",
  bad: "#f85149",
  selectedBg: "#0e639c",
  selectedFg: "#ffffff",
  inputBg: "#0e1014",
  /** Memory bar segments (match VS Code sidebar). */
  memWeights: "#3b82f6",
  memDraft: "#14b8a6",
  memKv: "#a855f7",
  memOverhead: "#64748b",
  memTrack: "#2a2e38",
} as const;
