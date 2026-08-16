/**
 * Client-side estimate of what fills the Copilot → llama.cpp context window.
 * Uses ~4 chars/token (same as estimateRequestTokens); good for bar shares, not exact IDs.
 */

export type ContextSegmentKey =
  | "tools"
  | "system"
  | "history"
  | "toolResults"
  | "request"
  | "free";

export interface ContextSegment {
  key: ContextSegmentKey;
  label: string;
  tokens: number;
}

export interface ContextBreakdown {
  /** Estimated tokens for each category (including free). */
  segments: ContextSegment[];
  /** Sum of used segments (excludes free). */
  usedTokens: number;
  /** Slot / context limit used for the free segment. */
  limitTokens: number;
  /** True when shares were scaled to match server prompt_tokens. */
  scaledToServer?: boolean;
}

const LABEL: Record<ContextSegmentKey, string> = {
  tools: "Tools",
  system: "System",
  history: "History",
  toolResults: "Tool results",
  request: "Request",
  free: "Free",
};

function contentChars(content: unknown): number {
  if (typeof content === "string") {
    return content.length;
  }
  if (content == null) {
    return 0;
  }
  if (Array.isArray(content)) {
    let n = 0;
    for (const part of content) {
      if (!part || typeof part !== "object") {
        continue;
      }
      const rec = part as { type?: string; text?: string };
      if (rec.type === "text" && typeof rec.text === "string") {
        n += rec.text.length;
      } else if (rec.type === "image_url" || rec.type === "input_video") {
        // Match estimateRequestTokens: do not count base64 bytes as tokens.
        n += rec.type === "input_video" ? 4096 : 1024;
      } else {
        try {
          n += JSON.stringify(part).length;
        } catch {
          // ignore
        }
      }
    }
    return n;
  }
  try {
    return JSON.stringify(content).length;
  } catch {
    return 0;
  }
}

export function messagesHaveImageParts(
  messages: Array<{ content?: unknown }>
): boolean {
  for (const m of messages) {
    if (!Array.isArray(m.content)) {
      continue;
    }
    for (const part of m.content) {
      if (part && typeof part === "object" && (part as { type?: string }).type === "image_url") {
        return true;
      }
    }
  }
  return false;
}

/**
 * Copilot-facing text when llama.cpp rejects a prompt for exceeding n_ctx.
 * Avoids the old canned “set Context Length ≥ 65536” when the slot is already
 * larger, and says so when raising context cannot fit the prompt.
 */
export function formatExceedContextError(opts: {
  nPrompt?: number;
  nCtx?: number;
  maxContext?: number;
  hasImages?: boolean;
  serverMessage?: string;
}): string {
  const nPrompt = opts.nPrompt && opts.nPrompt > 0 ? opts.nPrompt : undefined;
  const nCtx = opts.nCtx && opts.nCtx > 0 ? opts.nCtx : undefined;
  const maxContext = opts.maxContext && opts.maxContext > 0 ? opts.maxContext : undefined;
  const counts =
    nPrompt && nCtx ? ` (${nPrompt} tokens > ${nCtx} slot context)` : "";
  const lines: string[] = [`Context too small for this Copilot Chat request${counts}.`, ""];

  const overModelMax = !!(nPrompt && maxContext && nPrompt > maxContext);
  const overHardCap = !!(nPrompt && nPrompt > 262144 && (!maxContext || nPrompt > maxContext));
  const cannotFitByRaising = overModelMax || overHardCap;

  if (cannotFitByRaising) {
    lines.push(
      nPrompt && maxContext
        ? `This prompt is ${nPrompt} tokens; the model maximum is ${maxContext}. Raising Context Length will not fit it.`
        : `This prompt is larger than a llama.cpp context window can take. Raising Context Length will not fit it.`
    );
    lines.push("");
    lines.push("Shrink the request:");
    if (opts.hasImages) {
      lines.push(
        "• Pictures (screenshots / video frames) can cost many thousands of vision tokens each — send fewer, smaller, or more compressed images."
      );
    }
    lines.push("• Start a new chat to drop Copilot history and tool results.");
    lines.push("• Turn off tools you do not need for this turn.");
  } else if (nPrompt && nCtx && nPrompt > nCtx) {
    lines.push(`This prompt needs about ${nPrompt} tokens; the server slot is ${nCtx}.`);
    lines.push("");
    lines.push("Fix in Llama AIO sidebar:");
    const target = maxContext ? Math.min(nPrompt, maxContext) : nPrompt;
    lines.push(`1. Set Context Length to at least ${target}${maxContext ? ` (model max ${maxContext})` : ""}`);
    lines.push("2. Set Max Concurrent Predictions = 1 (parallel slots split the context)");
    lines.push('3. Click "Reload to apply changes"');
    if (opts.hasImages) {
      lines.push("");
      lines.push("If you attached pictures, shrinking them often helps more than raising context.");
    }
  } else {
    lines.push("Fix in Llama AIO sidebar:");
    lines.push("1. Raise Context Length (Copilot Chat usually needs 65536 or more)");
    lines.push("2. Set Max Concurrent Predictions = 1 (parallel slots split the context)");
    lines.push('3. Click "Reload to apply changes"');
  }

  if (opts.serverMessage) {
    lines.push("", `Server said: ${opts.serverMessage}`);
  }
  return lines.join("\n");
}

function messageChars(m: {
  role?: string;
  content?: unknown;
  tool_calls?: unknown;
}): number {
  let chars = (m.role?.length || 0) + 8;
  chars += contentChars(m.content);
  if (m.role === "assistant" && Array.isArray(m.tool_calls)) {
    for (const tc of m.tool_calls as Array<{
      function?: { name?: string; arguments?: string };
    }>) {
      chars += (tc.function?.name?.length || 0) + (tc.function?.arguments?.length || 0) + 24;
    }
  }
  return chars;
}

function charsToTokens(chars: number): number {
  return Math.max(0, Math.ceil(Math.max(0, chars) / 4));
}

/**
 * Classify OpenAI-style messages + tool defs into context-bar segments.
 * Last user message = Request; earlier user/assistant = History; tool role = Tool results.
 */
export function estimateContextBreakdown(
  messages: Array<{ role?: string; content?: unknown; tool_calls?: unknown }>,
  tools: unknown[] | undefined,
  limitTokens: number
): ContextBreakdown {
  let toolsChars = 0;
  if (tools?.length) {
    try {
      toolsChars = JSON.stringify(tools).length;
    } catch {
      toolsChars = 0;
    }
  }

  let systemChars = 0;
  let historyChars = 0;
  let toolResultChars = 0;
  let requestChars = 0;

  let lastUserIndex = -1;
  for (let i = 0; i < messages.length; i++) {
    if (messages[i]?.role === "user") {
      lastUserIndex = i;
    }
  }

  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    const c = messageChars(m);
    if (m.role === "system") {
      systemChars += c;
    } else if (m.role === "tool") {
      toolResultChars += c;
    } else if (m.role === "user" && i === lastUserIndex) {
      requestChars += c;
    } else if (m.role === "user" || m.role === "assistant") {
      historyChars += c;
    } else {
      historyChars += c;
    }
  }

  const toolsTok = charsToTokens(toolsChars);
  const systemTok = charsToTokens(systemChars);
  const historyTok = charsToTokens(historyChars);
  const toolResultsTok = charsToTokens(toolResultChars);
  const requestTok = charsToTokens(requestChars);
  const usedTokens = toolsTok + systemTok + historyTok + toolResultsTok + requestTok;
  const limit = Math.max(1, limitTokens || usedTokens);
  const freeTokens = Math.max(0, limit - usedTokens);

  const segments: ContextSegment[] = [
    { key: "tools", label: LABEL.tools, tokens: toolsTok },
    { key: "system", label: LABEL.system, tokens: systemTok },
    { key: "history", label: LABEL.history, tokens: historyTok },
    { key: "toolResults", label: LABEL.toolResults, tokens: toolResultsTok },
    { key: "request", label: LABEL.request, tokens: requestTok },
    { key: "free", label: LABEL.free, tokens: freeTokens },
  ];

  return { segments, usedTokens, limitTokens: limit };
}

/** Scale used segments so they sum to serverPromptTokens; recompute free from limit. */
export function scaleBreakdownToServerPrompt(
  breakdown: ContextBreakdown,
  serverPromptTokens: number,
  limitTokens?: number
): ContextBreakdown {
  const limit = Math.max(1, limitTokens ?? breakdown.limitTokens);
  const target = Math.max(0, serverPromptTokens);
  const usedKeys: ContextSegmentKey[] = [
    "tools",
    "system",
    "history",
    "toolResults",
    "request",
  ];
  const prevUsed = usedKeys.reduce((sum, key) => {
    const seg = breakdown.segments.find((s) => s.key === key);
    return sum + (seg?.tokens || 0);
  }, 0);

  const segments = breakdown.segments.map((seg) => {
    if (seg.key === "free") {
      return { ...seg, tokens: Math.max(0, limit - target) };
    }
    if (prevUsed <= 0) {
      // No prior estimate — put everything in request.
      return {
        ...seg,
        tokens: seg.key === "request" ? target : 0,
      };
    }
    const share = (seg.tokens || 0) / prevUsed;
    return { ...seg, tokens: Math.round(share * target) };
  });

  // Fix rounding so used segments sum exactly to target.
  const usedSum = segments
    .filter((s) => s.key !== "free")
    .reduce((a, b) => a + b.tokens, 0);
  const drift = target - usedSum;
  if (drift !== 0) {
    const adjust =
      segments.find((s) => s.key !== "free" && s.tokens > 0) ||
      segments.find((s) => s.key === "request");
    if (adjust) {
      adjust.tokens = Math.max(0, adjust.tokens + drift);
    }
  }

  const free = segments.find((s) => s.key === "free");
  if (free) {
    free.tokens = Math.max(0, limit - target);
  }

  return {
    segments,
    usedTokens: target,
    limitTokens: limit,
    scaledToServer: true,
  };
}
