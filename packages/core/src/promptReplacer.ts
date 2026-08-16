/**
 * Load and apply find/replace rules to Copilot system message text.
 * Ported from fuzzifikation/vLLM-Copilot (exact substring match, no regex).
 */

import * as fs from "fs/promises";
import * as path from "path";

export interface PromptReplacement {
  ruleName?: string;
  find: string;
  replace: string;
}

export interface PersonalityMeta {
  name: string;
  description: string;
}

/** Result of applying replacements to a single text blob. */
export interface ApplyResult {
  result: string;
  matchedRuleNames: string[];
}

/** Aggregate stats for one outbound chat request. */
export interface PromptReplacementStats {
  enabled: boolean;
  /** Estimated tokens for the full request before replacements (messages + tools). */
  tokensBefore: number;
  /** Estimated tokens after replacements. */
  tokensAfter: number;
  /** tokensBefore − tokensAfter */
  tokensSaved: number;
  /** tokensSaved / tokensBefore · 100 */
  pctSaved: number;
  matchedRuleNames: string[];
}

export async function loadPromptReplacements(filePath: string): Promise<PromptReplacement[]> {
  try {
    const absPath = path.resolve(filePath);
    const content = await fs.readFile(absPath, "utf-8");
    const trimmed = content.trim();
    if (!trimmed) {
      return [];
    }

    const parsed = JSON.parse(trimmed) as unknown;

    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      const obj = parsed as Record<string, unknown>;
      if (Array.isArray(obj.rules)) {
        return parseRules(obj.rules);
      }
      throw new Error(
        'Prompt replacements file with { meta, rules } format requires "rules" to be an array'
      );
    }

    if (Array.isArray(parsed)) {
      return parseRules(parsed);
    }

    throw new Error(
      "Prompt replacements file must contain a JSON array or a { meta, rules } object"
    );
  } catch (err) {
    if (err instanceof Error && "code" in err && (err as NodeJS.ErrnoException).code === "ENOENT") {
      return [];
    }
    throw err;
  }
}

function parseRules(parsed: unknown[]): PromptReplacement[] {
  const replacements: PromptReplacement[] = [];
  for (const entry of parsed) {
    if (typeof entry === "object" && entry !== null && "find" in entry && "replace" in entry) {
      const item = entry as Record<string, unknown>;
      if (typeof item.find === "string" && typeof item.replace === "string") {
        replacements.push({
          find: item.find,
          replace: item.replace,
          ruleName: typeof item.ruleName === "string" ? item.ruleName : undefined,
        });
      } else {
        throw new Error(
          `Each replacement entry must have "find" and "replace" as strings: ${JSON.stringify(entry).slice(0, 100)}`
        );
      }
    } else if (typeof entry === "object" && entry !== null) {
      throw new Error(
        `Each replacement entry must have "find" and "replace" properties: ${JSON.stringify(entry).slice(0, 100)}`
      );
    }
  }
  return replacements;
}

export function applyPromptReplacements(
  text: string,
  replacements: PromptReplacement[]
): ApplyResult {
  if (!replacements.length) {
    return { result: text, matchedRuleNames: [] };
  }

  const matchedRuleNames: string[] = [];
  let result = text;

  for (const { find, replace, ruleName } of replacements) {
    if (!find) {
      continue;
    }
    const count = result.split(find).length - 1;
    if (count > 0) {
      result = result.split(find).join(replace);
      if (ruleName) {
        matchedRuleNames.push(ruleName);
      }
    }
  }

  return { result, matchedRuleNames };
}

/** Rough char→token estimate (~4 chars/token), matching chatProvider. */
export function estimateTokensFromChars(chars: number): number {
  return Math.max(1, Math.ceil(Math.max(0, chars) / 4));
}

export function messageContentChars(messages: Array<{ role?: string; content?: unknown; tool_calls?: unknown }>): number {
  let chars = 0;
  for (const m of messages) {
    chars += (m.role?.length || 0) + 8;
    if (typeof m.content === "string") {
      chars += m.content.length;
    } else if (Array.isArray(m.content)) {
      for (const part of m.content) {
        if (!part || typeof part !== "object") {
          continue;
        }
        const rec = part as { type?: string; text?: string };
        if (rec.type === "text" && typeof rec.text === "string") {
          chars += rec.text.length;
        } else if (rec.type === "image_url") {
          // CLIP tokens vary; a few hundred is enough for the sidebar estimate.
          chars += 1024;
        }
      }
    }
    if (m.role === "assistant" && Array.isArray(m.tool_calls)) {
      for (const tc of m.tool_calls as Array<{ function?: { name?: string; arguments?: string } }>) {
        chars += (tc.function?.name?.length || 0) + (tc.function?.arguments?.length || 0) + 24;
      }
    }
  }
  return chars;
}

export function estimateRequestTokens(
  messages: Array<{ role?: string; content?: unknown; tool_calls?: unknown }>,
  tools?: unknown[]
): number {
  let chars = messageContentChars(messages);
  if (tools?.length) {
    try {
      chars += JSON.stringify(tools).length;
    } catch {
      // ignore
    }
  }
  return estimateTokensFromChars(chars);
}

/**
 * Apply replacements to system-role message contents only.
 * Returns a shallow-cloned message array and aggregate match names.
 */
export function applyReplacementsToSystemMessages<
  T extends { role?: string; content?: unknown },
>(messages: T[], replacements: PromptReplacement[]): { messages: T[]; matchedRuleNames: string[] } {
  if (!replacements.length) {
    return { messages, matchedRuleNames: [] };
  }
  const matched = new Set<string>();
  const out = messages.map((m) => {
    if (m.role !== "system" || typeof m.content !== "string" || !m.content) {
      return m;
    }
    const applied = applyPromptReplacements(m.content, replacements);
    for (const name of applied.matchedRuleNames) {
      matched.add(name);
    }
    if (applied.result === m.content) {
      return m;
    }
    return { ...m, content: applied.result };
  });
  return { messages: out, matchedRuleNames: [...matched] };
}

export function buildReplacementStats(opts: {
  enabled: boolean;
  tokensBefore: number;
  tokensAfter: number;
  matchedRuleNames: string[];
}): PromptReplacementStats {
  const tokensBefore = Math.max(0, opts.tokensBefore);
  const tokensAfter = Math.max(0, opts.tokensAfter);
  const tokensSaved = Math.max(0, tokensBefore - tokensAfter);
  const pctSaved =
    tokensBefore > 0 ? Math.round((tokensSaved / tokensBefore) * 1000) / 10 : 0;
  return {
    enabled: opts.enabled,
    tokensBefore,
    tokensAfter,
    tokensSaved,
    pctSaved,
    matchedRuleNames: opts.matchedRuleNames,
  };
}
