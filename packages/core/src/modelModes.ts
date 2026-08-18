/**
 * Copilot model modes (Think / No Think) via LanguageModelChatInformation.configurationSchema.
 * Mirrors fuzzifikation/vLLM-Copilot's reasoningEffort picker; params target llama-server
 * OpenAI-compatible chat completions (chat_template_kwargs + sampling).
 *
 * Qwen3 / 3.5 / 3.6: enable_thinking + temperature (Think General / Coding / No Think).
 * Qwen3.8: reasoning_effort (xhigh | medium | low) inside chat_template_kwargs.
 * Muse Glimmer: reasoning_strength (low | medium | high | xhigh); thinking cannot be off.
 * Gemma 4: enable_thinking on/off (template injects <|think|>); official top_k is 64.
 */

import type { ModelCapabilities } from "./ggufMetadata";

export interface ModelModeParams {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  presence_penalty?: number;
  repetition_penalty?: number;
  chat_template_kwargs?: Record<string, unknown>;
}

export interface ModelModeSet {
  /** Mode name → request overrides */
  modes: Record<string, ModelModeParams>;
  defaultMode: string;
  /** Short label for tooltips */
  familyLabel: string;
}

const QWEN_THINK_SAMPLING = {
  temperature: 1.0,
  top_p: 0.95,
  top_k: 20,
  min_p: 0.0,
  presence_penalty: 0.0,
  repetition_penalty: 1.0,
} as const;

const QWEN_NO_THINK: ModelModeParams = {
  chat_template_kwargs: {
    enable_thinking: false,
  },
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  min_p: 0.0,
  presence_penalty: 1.5,
  repetition_penalty: 1.0,
};

/** Official Qwen3.6 sampling tables (also used for Qwen3 / Qwen3.5 GGUFs). */
const QWEN3_THINKING_MODES: ModelModeSet = {
  familyLabel: "Qwen3",
  defaultMode: "Think (Coding)",
  modes: {
    "Think (General)": {
      chat_template_kwargs: {
        enable_thinking: true,
        preserve_thinking: true,
      },
      ...QWEN_THINK_SAMPLING,
    },
    "Think (Coding)": {
      chat_template_kwargs: {
        enable_thinking: true,
        preserve_thinking: true,
      },
      temperature: 0.6,
      top_p: 0.95,
      top_k: 20,
      min_p: 0.0,
      presence_penalty: 0.0,
      repetition_penalty: 1.0,
    },
    "No Think": QWEN_NO_THINK,
  },
};

/**
 * Qwen3.8 family (vLLM-Copilot Qwen-Qwen3.8-27B.json).
 * Labels expose the template's reasoning_effort values (xhigh / medium / low).
 * vLLM-Copilot's 27B draft only ships Deep (xhigh) + Balanced (medium); Low is
 * an official template level so Copilot can select it too.
 */
const QWEN38_THINKING_MODES: ModelModeSet = {
  familyLabel: "Qwen3.8",
  defaultMode: "Think (XHigh)",
  modes: {
    "Think (XHigh)": {
      chat_template_kwargs: {
        enable_thinking: true,
        preserve_thinking: true,
        reasoning_effort: "xhigh",
      },
      ...QWEN_THINK_SAMPLING,
    },
    "Think (Medium)": {
      chat_template_kwargs: {
        enable_thinking: true,
        preserve_thinking: true,
        reasoning_effort: "medium",
      },
      ...QWEN_THINK_SAMPLING,
    },
    "Think (Low)": {
      chat_template_kwargs: {
        enable_thinking: true,
        preserve_thinking: true,
        reasoning_effort: "low",
      },
      ...QWEN_THINK_SAMPLING,
    },
    "No Think": QWEN_NO_THINK,
  },
};

/** Gemma 4 / Muse Glimmer model-card sampling (top_k 64, not Qwen's 20). */
const GEMMA4_GLIMMER_SAMPLING = {
  temperature: 1.0,
  top_p: 0.95,
  top_k: 64,
} as const;

/**
 * Muse Glimmer — thinking is always on; depth is reasoning_strength.
 * https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF
 */
const GLIMMER_THINKING_MODES: ModelModeSet = {
  familyLabel: "Glimmer",
  defaultMode: "Think (High)",
  modes: {
    "Think (XHigh)": {
      chat_template_kwargs: { reasoning_strength: "xhigh" },
      ...GEMMA4_GLIMMER_SAMPLING,
    },
    "Think (High)": {
      chat_template_kwargs: { reasoning_strength: "high" },
      ...GEMMA4_GLIMMER_SAMPLING,
    },
    "Think (Medium)": {
      chat_template_kwargs: { reasoning_strength: "medium" },
      ...GEMMA4_GLIMMER_SAMPLING,
    },
    "Think (Low)": {
      chat_template_kwargs: { reasoning_strength: "low" },
      ...GEMMA4_GLIMMER_SAMPLING,
    },
  },
};

/**
 * Gemma 4 — binary thinking via enable_thinking (no effort ladder).
 * Official sampling is the same with thinking on or off.
 */
const GEMMA4_THINKING_MODES: ModelModeSet = {
  familyLabel: "Gemma 4",
  defaultMode: "Think",
  modes: {
    Think: {
      chat_template_kwargs: { enable_thinking: true },
      ...GEMMA4_GLIMMER_SAMPLING,
    },
    "No Think": {
      chat_template_kwargs: { enable_thinking: false },
      ...GEMMA4_GLIMMER_SAMPLING,
    },
  },
};

function haystack(caps?: ModelCapabilities, modelId?: string): string {
  return [caps?.architecture, caps?.name, caps?.path, modelId].filter(Boolean).join(" ").toLowerCase();
}

/** Qwen3.8 family — not Qwen3-8B (`qwen3-8b` / `qwen3-8B`). */
export function isQwen38Family(hay: string): boolean {
  const h = hay.toLowerCase();
  return /qwen3\.8/.test(h) || /qwen-?3\.8/.test(h) || /\bqwen38\b/.test(h) || /qwen3_8/.test(h);
}

/** Muse Glimmer (`general.architecture = muse-glimmer`). */
export function isGlimmerFamily(hay: string): boolean {
  const h = hay.toLowerCase();
  return /\bmuse-glimmer\b/.test(h) || /\bmuse_glimmer\b/.test(h) || /\bmuse glimmer\b/.test(h);
}

/** Gemma 4 language GGUFs — not Gemma 2/3, not the gemma4-assistant MTP sidecar. */
export function isGemma4Family(hay: string): boolean {
  const h = hay.toLowerCase();
  if (/gemma4-assistant|gemma4_assistant/.test(h)) {
    return false;
  }
  if (/\bmtp[-_]/.test(h) || /-mtp\.gguf/.test(h)) {
    return false;
  }
  return /\bgemma4\b/.test(h) || /gemma-4(?:\b|[._-])/.test(h);
}

/**
 * Resolve a curated mode set for the loaded model, or undefined when none apply.
 * Detection is intentionally narrow so we don't invent modes for every GGUF.
 */
export function resolveModelModes(
  caps?: ModelCapabilities,
  modelId?: string
): ModelModeSet | undefined {
  const h = haystack(caps, modelId);
  if (isQwen38Family(h)) {
    return QWEN38_THINKING_MODES;
  }
  if (isGlimmerFamily(h)) {
    return GLIMMER_THINKING_MODES;
  }
  if (isGemma4Family(h)) {
    return GEMMA4_THINKING_MODES;
  }
  // Qwen3 / 3.5 / 3.6 — arch may still be qwen2moe on some 3.5 GGUFs; prefer name.
  if (/\bqwen3/.test(h) || /qwen-?3(?:\.\d+)?/.test(h)) {
    return QWEN3_THINKING_MODES;
  }
  return undefined;
}

/** VS Code configurationSchema shape (not always in @types yet). */
export function buildModeConfigurationSchema(modeSet: ModelModeSet): {
  properties: Record<string, unknown>;
} {
  const modes = Object.keys(modeSet.modes);
  const defaultMode = modes.includes(modeSet.defaultMode) ? modeSet.defaultMode : modes[0];
  return {
    properties: {
      reasoningEffort: {
        type: "string",
        title: "Model Mode",
        enum: modes,
        enumItemLabels: modes,
        default: defaultMode,
        description: `${modeSet.familyLabel} thinking mode (chat_template_kwargs + sampling)`,
        group: "navigation",
      },
    },
  };
}

export function selectedModeName(
  modeSet: ModelModeSet,
  modelConfiguration: Record<string, unknown> | undefined
): string {
  const raw = modelConfiguration?.reasoningEffort;
  if (typeof raw === "string" && modeSet.modes[raw]) {
    return raw;
  }
  return modeSet.modes[modeSet.defaultMode]
    ? modeSet.defaultMode
    : Object.keys(modeSet.modes)[0];
}

export function resolveModeParams(
  modeSet: ModelModeSet,
  modelConfiguration: Record<string, unknown> | undefined
): { modeName: string; params: ModelModeParams } {
  const modeName = selectedModeName(modeSet, modelConfiguration);
  return { modeName, params: modeSet.modes[modeName] || {} };
}

/** Merge mode sampling + chat_template_kwargs into an OpenAI chat body. */
export function applyModeToRequestBody(
  body: Record<string, unknown>,
  params: ModelModeParams
): void {
  if (typeof params.temperature === "number") {
    body.temperature = params.temperature;
  }
  if (typeof params.top_p === "number") {
    body.top_p = params.top_p;
  }
  if (typeof params.top_k === "number") {
    body.top_k = params.top_k;
  }
  if (typeof params.min_p === "number") {
    body.min_p = params.min_p;
  }
  if (typeof params.presence_penalty === "number") {
    body.presence_penalty = params.presence_penalty;
  }
  if (typeof params.repetition_penalty === "number") {
    body.repetition_penalty = params.repetition_penalty;
  }
  if (params.chat_template_kwargs && typeof params.chat_template_kwargs === "object") {
    body.chat_template_kwargs = { ...params.chat_template_kwargs };
  }
}
