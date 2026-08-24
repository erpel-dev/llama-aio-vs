const FILE_SEARCH_TOOLS = new Set([
  "file_search",
  "glob_file_search",
  "glob",
  "find_by_name",
  "workspace_search",
]);

const GLOB_KEYS = new Set(["glob", "glob_pattern", "globpattern", "pattern", "query"]);

export function toolCallFingerprint(name: string, input: unknown): string {
  const n = (name || "").trim().toLowerCase();
  const obj =
    input && typeof input === "object" && !Array.isArray(input)
      ? (input as Record<string, unknown>)
      : { value: input };
  const norm: Record<string, unknown> = {};
  for (const key of Object.keys(obj).sort()) {
    const v = obj[key];
    if (v === undefined || v === null || v === "") {
      continue;
    }
    norm[key.toLowerCase()] = typeof v === "string" ? v.trim() : v;
  }
  return `${n}:${JSON.stringify(norm)}`;
}

/**
 * Fingerprints for assistant tool calls after the latest user message.
 * Earlier turns (including a read that has since fallen out of useful
 * context) are not blocked — only same-turn retries, which is the loop.
 */
export function fingerprintsSinceLastUserMessage(
  messages: Array<{
    role?: string;
    tool_calls?: Array<{ function?: { name?: string; arguments?: string } }>;
  }>
): Set<string> {
  let lastUser = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]!.role === "user") {
      lastUser = i;
      break;
    }
  }
  const seen = new Set<string>();
  for (let i = lastUser + 1; i < messages.length; i++) {
    const m = messages[i]!;
    if (m.role !== "assistant" || !Array.isArray(m.tool_calls)) {
      continue;
    }
    for (const tc of m.tool_calls) {
      const name = tc.function?.name || "";
      let input: unknown = {};
      try {
        input = JSON.parse(tc.function?.arguments || "{}");
      } catch {
        input = { raw: tc.function?.arguments || "" };
      }
      seen.add(toolCallFingerprint(name, input));
    }
  }
  return seen;
}

/**
 * A glob with a literal basename and no wildcards (e.g. star-star/llama) only
 * matches a file of that exact name. Tiny models loop on that after Copilot
 * returns no matches. Expand to a contains-glob.
 */
export function expandLiteralFileGlob(pattern: string): string | undefined {
  const p = pattern.trim();
  const m = /^(?:\*\*\/)?([^/*?\[\]]+)$/.exec(p);
  if (!m) {
    return undefined;
  }
  const base = m[1]!;
  if (base.includes(".") || base.length < 2) {
    return undefined;
  }
  return `**/*${base}*`;
}

export function rewriteToolInput(name: string, input: unknown): unknown {
  const n = (name || "").trim().toLowerCase();
  if (!FILE_SEARCH_TOOLS.has(n) && !n.endsWith("_file_search")) {
    return input;
  }
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    return input;
  }
  const obj = input as Record<string, unknown>;
  let changed = false;
  const next: Record<string, unknown> = { ...obj };
  for (const [key, value] of Object.entries(obj)) {
    if (typeof value !== "string" || !GLOB_KEYS.has(key.toLowerCase())) {
      continue;
    }
    const expanded = expandLiteralFileGlob(value);
    if (expanded && expanded !== value) {
      next[key] = expanded;
      changed = true;
    }
  }
  return changed ? next : input;
}

export function duplicateToolCallHint(name: string, input: unknown): string {
  const glob =
    input && typeof input === "object" && !Array.isArray(input)
      ? Object.values(input as Record<string, unknown>).find((v) => typeof v === "string")
      : undefined;
  const shown = typeof glob === "string" && glob ? glob : name;
  return `Skipped repeating \`${name}\` (\`${shown}\`); that call already ran this turn. Broaden the glob (e.g. **/*llama* not **/llama) or grep for a symbol instead.`;
}
