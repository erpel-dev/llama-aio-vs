/**
 * Copilot model modes (Think / No Think) via LanguageModelChatInformation.configurationSchema.
 * Mirrors fuzzifikation/vLLM-Copilot's reasoningEffort picker; params target llama-server
 * OpenAI-compatible chat completions (chat_template_kwargs.enable_thinking, sampling).
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
      temperature: 1.0,
      top_p: 0.95,
      top_k: 20,
      min_p: 0.0,
      presence_penalty: 0.0,
      repetition_penalty: 1.0,
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
    "No Think": {
      chat_template_kwargs: {
        enable_thinking: false,
      },
      temperature: 0.7,
      top_p: 0.8,
      top_k: 20,
      min_p: 0.0,
      presence_penalty: 1.5,
      repetition_penalty: 1.0,
    },
  },
};

function haystack(caps?: ModelCapabilities, modelId?: string): string {
  return [caps?.architecture, caps?.name, caps?.path, modelId].filter(Boolean).join(" ").toLowerCase();
}

/**
 * Resolve a curated mode set for the loaded model, or undefined when none apply.
 * Detection is intentionally narrow (Qwen3*) so we don't invent modes for every GGUF.
 */
export function resolveModelModes(
  caps?: ModelCapabilities,
  modelId?: string
): ModelModeSet | undefined {
  const h = haystack(caps, modelId);
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
