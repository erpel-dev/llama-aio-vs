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
  try {
    return JSON.stringify(content).length;
  } catch {
    return 0;
  }
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
