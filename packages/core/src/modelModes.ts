/**
 * Copilot model modes (Think / No Think) via LanguageModelChatInformation.configurationSchema.
 * Mirrors fuzzifikation/vLLM-Copilot's reasoningEffort picker; params target llama-server
 * OpenAI-compatible chat completions (chat_template_kwargs + sampling).
 *
 * Author-recommended request presets. See docs/model-profiles.md for sources,
 * supported checkpoints, backend requirements, and verification limits.
 * Qwen3.8 dense: reasoning_effort (xhigh | medium | low) inside chat_template_kwargs.
 * Qwen3.8-Flash-Next (qwen4exp): same effort ladder; llama.cpp injects it via
 * chat_template_kwargs (not a top-level body field — that is vLLM-only).
 * Muse Glimmer: reasoning_strength (low | medium | high | xhigh); thinking cannot be off.
 * Gemma 4: enable_thinking on/off (template injects <|think|>); official top_k is 64.
 */

import type { ModelCapabilities } from "./ggufMetadata";
import type { RemoteServerInfo } from "./remoteServer";

export interface ModelModeParams {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  repetition_penalty?: number;
  chat_template_kwargs?: Record<string, unknown>;
}

export interface ModelModeSet {
  /** Mode name → request overrides */
  modes: Record<string, ModelModeParams>;
  defaultMode: string;
  /** Short label for tooltips */
  familyLabel: string;
  /** Reasoning must survive assistant/tool turns on hosts with thinking parts. */
  preserveReasoning?: boolean;
}

const QWEN_THINK_SAMPLING = {
  temperature: 1.0,
  top_p: 0.95,
  top_k: 20,
  min_p: 0.0,
  presence_penalty: 0.0,
  frequency_penalty: 0.0,
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
  frequency_penalty: 0.0,
  repetition_penalty: 1.0,
};

/** Qwen3.5/3.6 general and coding recipes differ from original Qwen3. */
const QWEN35_THINKING_MODES: ModelModeSet = {
  familyLabel: "Qwen3.5 / 3.6",
  defaultMode: "Think (Coding)",
  modes: {
    "Think (General)": {
      chat_template_kwargs: {
        enable_thinking: true,
        preserve_thinking: true,
      },
      ...QWEN_THINK_SAMPLING,
      presence_penalty: 1.5,
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
      frequency_penalty: 0.0,
      repetition_penalty: 1.0,
    },
    "No Think": QWEN_NO_THINK,
  },
};

const QWEN3_THINKING_MODES: ModelModeSet = {
  familyLabel: "Qwen3",
  defaultMode: "Think",
  modes: {
    Think: { ...QWEN_THINK_SAMPLING, temperature: 0.6, chat_template_kwargs: { enable_thinking: true } },
    "No Think": { ...QWEN_NO_THINK, presence_penalty: 0.0 },
  },
};

/**
 * Qwen3.8 family (vLLM-Copilot Qwen-Qwen3.8-27B.json) and Qwen3.8-Flash-Next
 * (Qwen-Qwen3.8-Flash-Next.json). Labels expose the template's reasoning_effort
 * values (xhigh / medium / low).
 *
 * vLLM-Copilot's presets only ship Deep (xhigh) + Balanced (medium). Low is an
 * official template level (unknown values raise), so Copilot can select it too.
 *
 * llama.cpp / Unsloth pass `reasoning_effort` inside chat_template_kwargs.
 * vLLM-Copilot puts it at the top level of the vLLM extra body — that does not
 * reach llama-server's Jinja context, so we keep it in kwargs here.
 */
const QWEN38_EFFORT_MODES: Record<string, ModelModeParams> = {
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
};

const QWEN38_THINKING_MODES: ModelModeSet = {
  familyLabel: "Qwen3.8",
  defaultMode: "Think (XHigh)",
  modes: QWEN38_EFFORT_MODES,
};

const QWEN4EXP_THINKING_MODES: ModelModeSet = {
  familyLabel: "Qwen3.8 Flash-Next",
  defaultMode: "Think (XHigh)",
  modes: QWEN38_EFFORT_MODES,
};

/** Gemma 4 / Muse Glimmer model-card sampling (top_k 64, not Qwen's 20). */
const GEMMA4_GLIMMER_SAMPLING = {
  min_p: 0.0,
  presence_penalty: 0.0,
  frequency_penalty: 0.0,
  repetition_penalty: 1.0,
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

// Disable undocumented samplers rather than inherit llama-server or user values
// that alter an author's recipe. top_k <= 0 disables top-k in llama.cpp; -1
// also works with vLLM/SGLang and is Spark's published generation default.
const OPEN_SAMPLING: ModelModeParams = {
  temperature: 1.0, top_p: 1.0, top_k: -1, min_p: 0.0,
  presence_penalty: 0.0, frequency_penalty: 0.0, repetition_penalty: 1.0,
};

function binaryModes(familyLabel: string, sampling: ModelModeParams): ModelModeSet {
  return {
    familyLabel, defaultMode: "Think",
    modes: {
      Think: { ...sampling, chat_template_kwargs: { enable_thinking: true } },
      "No Think": { ...sampling, chat_template_kwargs: { enable_thinking: false } },
    },
  };
}

function effortModes(
  familyLabel: string,
  efforts: readonly string[],
  defaultEffort: string,
  sampling: ModelModeParams,
  kwargs: Record<string, unknown> = {}
): ModelModeSet {
  const label = (effort: string) => `Think (${effort === "xhigh" ? "XHigh" : effort[0].toUpperCase() + effort.slice(1)})`;
  return {
    familyLabel, defaultMode: label(defaultEffort),
    modes: Object.fromEntries(efforts.map(effort => [label(effort), {
      ...sampling, chat_template_kwargs: { ...kwargs, reasoning_effort: effort },
    }])),
  };
}

function fixedMode(familyLabel: string, mode: "Think" | "No Think", sampling: ModelModeParams): ModelModeSet {
  return { familyLabel, defaultMode: mode, modes: { [mode]: { ...sampling, chat_template_kwargs: {} } } };
}

const SPARK_SAMPLING = { ...OPEN_SAMPLING, top_p: 0.95 };
const SPARK25_MODES = binaryModes("Spark X2.5", SPARK_SAMPLING);
const SPARK26_MODES = effortModes("Spark X2.6", ["high", "low"], "high", SPARK_SAMPLING, { enable_thinking: true });
SPARK26_MODES.preserveReasoning = true;
SPARK26_MODES.modes["No Think"] = { ...SPARK_SAMPLING, chat_template_kwargs: { enable_thinking: false } };

const GPT_OSS_MODES = effortModes("GPT-OSS", ["high", "medium", "low"], "medium", OPEN_SAMPLING);
const GLM53_MODES = effortModes("GLM-5.3-Flash", ["max", "high", "low"], "max",
  { ...OPEN_SAMPLING, top_p: 0.95 }, { clear_thinking: true });
const GLM53_FULL_MODES: ModelModeSet = { ...GLM53_MODES, familyLabel: "GLM-5.3" };

const MINICPM5_MODES = binaryModes("MiniCPM5 2B", { ...OPEN_SAMPLING, top_p: 0.95 });
// FrogNano's author evaluates at temperature 0.6 and repeat penalty 1.
// Its inherited Qwen3.5 template supports the binary thinking switch.
const FROGNANO_MODES = binaryModes("FrogNano 4B 2609", { ...OPEN_SAMPLING, temperature: 0.6 });
/** Nanbeige4.3-33B: author agent/chat recipes, not distinct reasoning depths. */
const NANBEIGE43_MODES: ModelModeSet = {
  familyLabel: "Nanbeige4.3 33B", defaultMode: "Think (Agent)", preserveReasoning: true,
  modes: {
    "Think (Agent)": { ...OPEN_SAMPLING, temperature: 1, top_p: 0.95,
      chat_template_kwargs: { enable_thinking: true, preserve_thinking: true, tool_call_format: "xml" } },
    "Think (Chat)": { ...OPEN_SAMPLING, temperature: 0.6, top_p: 0.95,
      chat_template_kwargs: { enable_thinking: true, preserve_thinking: false, tool_call_format: "xml" } },
    // No separate off-mode sampler is published; use the chat recipe.
    "No Think": { ...OPEN_SAMPLING, temperature: 0.6, top_p: 0.95,
      chat_template_kwargs: { enable_thinking: false, preserve_thinking: false, tool_call_format: "xml" } },
  },
};
const MELLUM21_MODES = fixedMode("Mellum 2.1 Thinking", "Think", {
  ...OPEN_SAMPLING, temperature: 0.6, top_p: 0.95, top_k: 20,
});
const MIMO26_DISTILL_MODES = fixedMode("MiMo V2.6 Distill Qwen 9B", "Think", {
  ...OPEN_SAMPLING, temperature: 0.6, top_p: 0.95, top_k: 20,
});
const FLINT_MODES: ModelModeSet = {
  ...fixedMode("Flint", "Think", { ...OPEN_SAMPLING, top_p: 0.95, top_k: 20 }),
  preserveReasoning: true,
};
const SAGE15_MODES: ModelModeSet = {
  familyLabel: "Sage A1.5", defaultMode: "Think",
  modes: { Think: { ...OPEN_SAMPLING, top_p: 0.95, top_k: 20, presence_penalty: 1.5,
    chat_template_kwargs: { enable_thinking: true, preserve_thinking: true } } },
};
const SALUKI_MODES: ModelModeSet = {
  familyLabel: "Underdog Saluki 27B", defaultMode: "Think",
  modes: {
    Think: { ...OPEN_SAMPLING, temperature: 0.6, top_p: 0.95, top_k: 20,
      chat_template_kwargs: { enable_thinking: true } },
    "No Think": { ...OPEN_SAMPLING, temperature: 0, top_p: 0.95, top_k: 20,
      chat_template_kwargs: { enable_thinking: false } },
  },
};
function deepSeek41Modes(familyLabel: string, topP: number): ModelModeSet {
  const sampling = { ...OPEN_SAMPLING, top_p: topP };
  return {
    familyLabel, defaultMode: "Think (High)", preserveReasoning: true,
    modes: {
      ...Object.fromEntries([["Max", 100], ["High", 75], ["Low", 50]].map(([name, budget]) => [
        `Think (${name})`, { ...sampling, chat_template_kwargs: { enable_thinking: true, reasoning_effort: budget } },
      ])),
      "No Think": { ...sampling, chat_template_kwargs: { enable_thinking: false } },
    },
  };
}
// Both cards permit top-p 1; Strata/Maya use this undated ID for 1017 too.
const DEEPSEEK41_MODES = deepSeek41Modes("DeepSeek V4.1 Flash", 1);
const DEEPSEEK41_1017_MODES = deepSeek41Modes("DeepSeek V4.1 Flash 1017", 1);

const QWEN_CODER_MODES = fixedMode("Qwen3-Coder Instruct", "No Think", {
  ...OPEN_SAMPLING, temperature: 0.7, top_p: 0.8, top_k: 20, repetition_penalty: 1.05,
});
const QWEN25_STARTER_MODES = fixedMode("Qwen2.5 0.5B Instruct", "No Think", {
  ...OPEN_SAMPLING, temperature: 0.7, top_p: 0.8, top_k: 20, repetition_penalty: 1.1,
});
const QWEN_CODER_NEXT_MODES = fixedMode("Qwen3-Coder-Next", "No Think", {
  ...OPEN_SAMPLING, top_p: 0.95, top_k: 40,
});
const QWEN3_INSTRUCT_MODES = fixedMode("Qwen3 Instruct 2507", "No Think", {
  ...OPEN_SAMPLING, temperature: 0.7, top_p: 0.8, top_k: 20,
});
const QWEN3_THINKING_ONLY_MODES = fixedMode("Qwen3 Thinking 2507", "Think", {
  ...OPEN_SAMPLING, temperature: 0.6, top_p: 0.95, top_k: 20,
});
// This distillation names the Qwen3.8 teacher, but retains a Qwen3.6 template.
// It has no reasoning_effort ladder; the author recommends .6/.95/20.
const QWEN38_DISTILL_MODES: ModelModeSet = {
  familyLabel: "Qwen3.8 Distill 35B-A3B", defaultMode: "Think",
  modes: { Think: { ...OPEN_SAMPLING, temperature: 0.6, top_p: 0.95, top_k: 20,
    chat_template_kwargs: { enable_thinking: true } } },
};

// These verified Ornith checkpoints use Qwen3.5 templates and recipes. Other
// sizes or revisions must not be assigned the same base family by brand alone.
const ORNITH15_MODES: ModelModeSet = { ...QWEN35_THINKING_MODES, familyLabel: "Ornith 1.5 (Qwen)" };
const TIEL_CODER_MODES = binaryModes("Tiel-Coder 35B-A3B", {
  ...OPEN_SAMPLING, temperature: 0.6, top_p: 0.95, top_k: 20,
});
TIEL_CODER_MODES.modes["No Think"] = QWEN_NO_THINK;

function nemotronNanoModes(familyLabel: string, reasoningTopP: number): ModelModeSet {
  return {
    familyLabel, defaultMode: "Think (Coding)",
    modes: {
      "Think (General)": { ...OPEN_SAMPLING, top_p: reasoningTopP, chat_template_kwargs: { enable_thinking: true } },
      "Think (Coding)": { ...OPEN_SAMPLING, temperature: 0.6, top_p: 0.95, chat_template_kwargs: { enable_thinking: true } },
      "No Think": { ...OPEN_SAMPLING, temperature: 0.6, top_p: 0.95, chat_template_kwargs: { enable_thinking: false } },
    },
  };
}
const NEMOTRON_NANO4_MODES = nemotronNanoModes("Nemotron 3 Nano 4B", 0.95);
const NEMOTRON_NANO30_MODES = nemotronNanoModes("Nemotron 3 Nano 30B", 1.0);
const NEMOTRON_BOLT_MODES = binaryModes("Nemotron 3.5 Bolt E4B", GEMMA4_GLIMMER_SAMPLING);

const LFM25_INSTRUCT_MODES = fixedMode("LFM2.5 Instruct", "No Think", {
  ...OPEN_SAMPLING, temperature: 0.1, top_k: 50, repetition_penalty: 1.05,
});
const LFM25_DENSE_MODES = fixedMode("LFM2.5 2.6B", "Think", {
  ...OPEN_SAMPLING, temperature: 0.1, top_k: 50, repetition_penalty: 1.1,
});
const LFM25_MOE_MODES = fixedMode("LFM2.5 24B-A2B", "Think", {
  ...OPEN_SAMPLING, temperature: 0.1, top_k: 50, repetition_penalty: 1.05,
});
// LFM3's General and Math/STEM settings are recipes, not reasoning depths.
// Keep the generation config's repeat penalty in both recipes. The template
// clears old reasoning while retaining the current assistant/tool turn.
const LFM3_GENERAL_SAMPLING = { ...OPEN_SAMPLING, temperature: 0.1, top_k: 50, repetition_penalty: 1.1 };
const LFM3_MODES: ModelModeSet = {
  familyLabel: "LFM3 1.7B", defaultMode: "Think (General)", preserveReasoning: true,
  modes: {
    "Think (General)": { ...LFM3_GENERAL_SAMPLING,
      chat_template_kwargs: { enable_thinking: true, preserve_thinking: false } },
    "Think (Math / STEM)": { ...LFM3_GENERAL_SAMPLING, temperature: 1, top_p: 0.95, top_k: 20,
      chat_template_kwargs: { enable_thinking: true, preserve_thinking: false } },
    "No Think": { ...LFM3_GENERAL_SAMPLING,
      chat_template_kwargs: { enable_thinking: false, preserve_thinking: false } },
  },
};
const BONSAI2_MODES = effortModes("Ternary Bonsai 2 27B", ["xhigh", "medium"], "xhigh",
  { ...QWEN_THINK_SAMPLING, min_p: 0.05 }, { enable_thinking: true });
BONSAI2_MODES.modes["No Think"] = QWEN_NO_THINK;

/** Only actual model identity, excluding parent directories named for other models. */
function haystack(caps?: ModelCapabilities, modelId?: string): string {
  const basename = (value?: string) => value?.split(/[/\\]/).pop();
  return [caps?.name, basename(caps?.path), basename(modelId)].filter(Boolean).join(" ").toLowerCase()
    // GGUF publishers commonly prefix names with `author_`. Treat that as a
    // word boundary, while preserving numeric versions such as Qwen3_8.
    .replace(/(?<!\d)_|_(?!\d)/g, "-");
}

function namedProfile(h: string): ModelModeSet | undefined {
  if (/(?:^|[-_\s])nanbeige[-_ ]?4[._]3[-_ ]33b(?=$|[.\s]|[-_](?:gguf|(?:ud[-_ ])?(?:iq|q|fp|bf|mxfp|gsq)[\w.-]*)(?:$|[-_.\s]))/.test(h)) return NANBEIGE43_MODES;
  if (/\bqwen[-_ ]?2[._]5[-_ ]0[._]5b[-_ ]instruct(?:\b|_)/.test(h)) return QWEN25_STARTER_MODES;
  if (/\btiel[-_ ]coder[-_ ]35b[-_ ]a3b(?:\b|_)/.test(h)) return TIEL_CODER_MODES;
  if (/\bminicpm[-_ ]?5[-_ ]2(?:[._]6)?b(?:\b|_)/.test(h)) return MINICPM5_MODES;
  if (/\bfrognano[-_ ]4b[-_ ]2609(?:\b|_)/.test(h)) return FROGNANO_MODES;
  if (/\bmellum[-_ ]?2[._]1[-_ ]12b[-_ ]a2[._]5b[-_ ]thinking(?:\b|_)/.test(h)) return MELLUM21_MODES;
  if (/\bmimo[-_ ]v2[._]6[-_ ]distill[-_ ]qwen[-_ ]9b(?:\b|_)/.test(h)) return MIMO26_DISTILL_MODES;
  if (/\b(?:underdog[-_ ])?saluki[-_ ]27b[-_ ]1[._]0(?:\b|_)/.test(h)) return SALUKI_MODES;
  if (/\bflint(?:$|[-_.\s])/.test(h)) return FLINT_MODES;
  if (/\bsage[-_ ]a1[._]5(?:\b|_)/.test(h)) return SAGE15_MODES;
  if (/\bdeepseek[-_ ]v4[._]1[-_ ]flash[-_ ]1017(?:\b|_)/.test(h)) return DEEPSEEK41_1017_MODES;
  if (/\bdeepseek[-_ ]v4[._]1[-_ ]flash[-_ ]ist[-_ ]das[-_ ](?:1[._]75|2[._](?:00|25))bpw(?:\b|_)/.test(h)) return DEEPSEEK41_1017_MODES;
  if (/\bdeepseek[-_ ]v4[._]1[-_ ]flash(?=$|[.\s]|[-_](?:gguf|(?:ud[-_ ])?(?:iq|q|fp|bf|mxfp|gsq)[\w.-]*)(?:$|[-_.\s]))/.test(h)) return DEEPSEEK41_MODES;
  if (/\bqwen3[._]8[-_ ]35b[-_ ]a3b[-_ ]distill(?:\b|_)/.test(h)) return QWEN38_DISTILL_MODES;
  if (/\bspark[-_ ]?x?2[._]6(?=$|[-_.\s/\\])/.test(h)) return SPARK26_MODES;
  if (/\bspark[-_ ]?x?2[._]5(?=$|[-_.\s/\\])/.test(h)) return SPARK25_MODES;
  if (/\bgpt[-_ ]oss(?:\b|_)/.test(h)) return GPT_OSS_MODES;
  if (/\bglm[-_ ]?5[._]3[-_ ]flash(?:\b|_)/.test(h)) return GLM53_MODES;
  if (/\bglm[-_ ]?5[._]3(?=$|[.\s]|[-_](?:gguf|(?:ud[-_ ])?(?:iq|q|fp|bf|mxfp)[\w.-]*)(?:$|[-_.\s]))/.test(h)) return GLM53_FULL_MODES;
  if (/\b(?:ternary[-_ ])?bonsai[-_ ]?2[-_ ]27b(?:\b|_)/.test(h)) return BONSAI2_MODES;
  if (/\bornith[-_ ]1[._]5[-_ ](?:9b|35b(?:[-_ ]a3b)?|122b)(?:\b|_)/.test(h)) return ORNITH15_MODES;
  if (/\bnemotron[-_ ]3[._]5[-_ ]bolt[-_ ]e4b(?:\b|_)/.test(h)) return NEMOTRON_BOLT_MODES;
  if (/\bnemotron[-_ ]3[-_ ]nano[-_ ]4b(?:\b|_)/.test(h)) return NEMOTRON_NANO4_MODES;
  if (/\bnemotron[-_ ]3[-_ ]nano[-_ ]30b[-_ ]a3b(?:\b|_)/.test(h)) return NEMOTRON_NANO30_MODES;
  if (/\blfm2[._]5[-_ ]1[._]2b[-_ ]instruct(?:\b|_)/.test(h)) return LFM25_INSTRUCT_MODES;
  if (/\blfm2[._]5[-_ ]230m(?:\b|_)/.test(h)) return LFM25_INSTRUCT_MODES;
  if (/\blfm2[._]5[-_ ]2[._]6b(?:\b|_)/.test(h)) return LFM25_DENSE_MODES;
  if (/\blfm2[._]5[-_ ]24b[-_ ]a2b(?:\b|_)/.test(h)) return LFM25_MOE_MODES;
  if (/\blfm[-_ ]?3[-_ ]1[._]7b(?=$|[.\s]|[-_](?:gguf|qad|mlx|onnx|(?:ud[-_ ])?(?:iq|q|fp|bf|mxfp|nvfp)[\w.-]*)(?:$|[-_.\s]))/.test(h)) return LFM3_MODES;
  if (/\bqwen3[-_ ]coder[-_ ]next(?:\b|_)/.test(h)) return QWEN_CODER_NEXT_MODES;
  if (/\bqwen3[-_ ]coder[-_ ](?:30b[-_ ]a3b|480b[-_ ]a35b)[-_ ]instruct(?:\b|_)/.test(h)) return QWEN_CODER_MODES;
  if (/\bqwen3[-_ ].*?[-_ ]instruct[-_ ]2507(?:\b|_)/.test(h)) return QWEN3_INSTRUCT_MODES;
  if (/\bqwen3[-_ ].*?[-_ ]thinking[-_ ]2507(?:\b|_)/.test(h)) return QWEN3_THINKING_ONLY_MODES;
  if (isQwen4expFamily(h)) return QWEN4EXP_THINKING_MODES;
  if (isQwen38Family(h)) return QWEN38_THINKING_MODES;
  if (isGlimmerFamily(h)) return GLIMMER_THINKING_MODES;
  if (isGemma4Family(h)) return GEMMA4_THINKING_MODES;
  if (/\bqwen[-_ ]?3[._][56](?=$|[-_.\s/\\])/.test(h)) return QWEN35_THINKING_MODES;
  if (/\bqwen3[-_ ]coder(?:\b|_)/.test(h)) return undefined;
  if (/\bqwen[-_ ]?3(?![._]\d)(?:\b|_)/.test(h)) return QWEN3_THINKING_MODES;
  return undefined;
}

/**
 * Qwen3.8-Flash-Next / qwen4exp — must run before {@link isQwen38Family}
 * because filenames contain `Qwen3.8`.
 */
export function isQwen4expFamily(hay: string): boolean {
  const h = hay.toLowerCase();
  // Strata's Swift 1.5 fine-tune IDs omit the base model name entirely.
  if (/\bswift[-_ ]1[._]5(?=$|[-_.\s/\\])/.test(h)) {
    // A dense Swift quant names its Qwen3.8-27B base explicitly. Only the
    // shortened Strata alias should imply Flash-Next.
    return !/qwen[-_ ]?3[._]8[-_ ]27b(?:\b|_)/.test(h);
  }
  if (/qwen4exp/.test(h) || /qwen4_exp/.test(h) || /qwen4-exp/.test(h)) {
    return true;
  }
  if (!/qwen/.test(h)) {
    return false;
  }
  return (
    /qwen[-_ ]?3[._-]8[-_. ]flash/.test(h) ||
    /qwen-?3\.8[-_.]?flash/.test(h) ||
    /qwen38[-_.]?flash/.test(h) ||
    /flash[-_]?next/.test(h)
  );
}

/** Qwen3.8 dense family — not Qwen3-8B and not Flash-Next. */
export function isQwen38Family(hay: string): boolean {
  if (isQwen4expFamily(hay)) {
    return false;
  }
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
  return /\bgemma4(?:_unified)?\b/.test(h) || /gemma-4(?:\b|[._-])/.test(h);
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
  const arch = (caps?.architecture || "").toLowerCase();
  if (/^(?:dflash|gemma4[-_]assistant)$/.test(arch) || profileVariantExcluded(h)) return undefined;
  const named = namedProfile(h);
  if (named) return named;
  // An explicitly named, unverified variant must not inherit a misleading
  // profile just because its base architecture is supported.
  if (/\b(?:frognano|spark[-_ ]?x?\d|ornith[-_ ]\d|lfm(?:2[._]\d|[-_ ]?3)|nemotron[-_ ]\d|(?:ternary[-_ ])?bonsai[-_ ]?\d|qwen3[-_ ]coder)/.test(h)) return undefined;
  if (/^gpt[-_]oss$/.test(arch)) return GPT_OSS_MODES;
  if (arch === "qwen4exp" || arch === "qwen4_exp") return QWEN4EXP_THINKING_MODES;
  if (/^qwen(?:35|3_5)/.test(arch)) return QWEN35_THINKING_MODES;
  if (/^qwen3(?:moe)?$/.test(arch)) return QWEN3_THINKING_MODES;
  // Spark X2.5 and X2.6 share spark2_5; architecture alone cannot select
  // their different controls (some published GGUF names are just "Dense F32").
  if (isGlimmerFamily(arch)) return GLIMMER_THINKING_MODES;
  if (isGemma4Family(arch)) return GEMMA4_THINKING_MODES;
  return undefined;
}

function profileVariantExcluded(h: string): boolean {
  return /(?:^|[\s])(?:mmproj|mtp)[-_.]/.test(h)
    || /(?:^|[-_.\s])mmproj(?:$|[-_.\s])/.test(h)
    || /-mtp\.gguf(?:$|\s)/.test(h)
    || /(?:^|[-_.\s])dflash\d*(?:$|[-_.\s])/.test(h)
    || /(?:^|[-_.\s])dspark\d*(?:$|[-_.\s])/.test(h)
    || /(?:^|[-_.\s])draft(?:$|[-_.\s])/.test(h)
    || /(?:^|[-_.\s])base(?:$|[-_.\s])/.test(h);
}

/** Pending local selections must not change the running server's request recipe. */
export function resolveLocalModelModes(
  caps: ModelCapabilities | undefined,
  modelId: string | undefined,
  loadedModelPath?: string
): ModelModeSet | undefined {
  if (loadedModelPath && loadedModelPath !== caps?.path) {
    return resolveRemoteModelModes({ modelPath: loadedModelPath, modelId });
  }
  return resolveModelModes(caps, modelId);
}

/** Shared by remote chat and the sidebar; never consult local model settings. */
export function resolveRemoteModelModes(
  info: Pick<RemoteServerInfo, "modelAlias" | "modelPath" | "modelName" | "modelId" | "modelAliases"> | undefined
): ModelModeSet | undefined {
  if (!info) return undefined;
  const identities = [info.modelAlias, info.modelPath, info.modelName, info.modelId];
  // Canonical Base/draft identities take precedence over generic chat aliases.
  if (identities.some(identity => identity && profileVariantExcluded(haystack(undefined, identity)))) return undefined;
  for (const identity of identities) {
    const profile = resolveModelModes(undefined, identity);
    if (profile) return profile;
  }
  const aliases = (info.modelAliases ?? []).map(alias => resolveModelModes(undefined, alias))
    .filter((profile): profile is ModelModeSet => !!profile);
  // An arbitrary alias list may name different families. Leave sampling free
  // if it contradicts itself instead of choosing whichever alias appeared first.
  return aliases.length && aliases.every(profile => profile.familyLabel === aliases[0].familyLabel)
    ? aliases[0] : undefined;
}

// Copilot CLI filters reasoningEffort against its standard effort values. Keep
// our preset names as labels only; using them as enum values hides the picker.
const MODE_EFFORT_VALUES: Record<string, string> = {
  "Think (Agent)": "high",
  "Think (Chat)": "medium",
  "Think (Max)": "max",
  "Think (XHigh)": "xhigh",
  "Think (High)": "high",
  "Think (Medium)": "medium",
  "Think (Low)": "low",
  "Think (General)": "high",
  "Think (Math / STEM)": "medium",
  "Think (Coding)": "medium",
  Think: "high",
  "No Think": "none",
};

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
        enum: modes.map(name => MODE_EFFORT_VALUES[name] ?? name),
        enumItemLabels: modes,
        default: MODE_EFFORT_VALUES[defaultMode] ?? defaultMode,
        description: `${modeSet.familyLabel} request preset`,
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
    // Accept names saved by earlier versions, including existing chat sessions.
    return raw;
  }
  const mapped = Object.keys(modeSet.modes).find(name => MODE_EFFORT_VALUES[name] === raw);
  if (mapped) {
    return mapped;
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
  if (typeof params.frequency_penalty === "number") {
    body.frequency_penalty = params.frequency_penalty;
  }
  if (typeof params.repetition_penalty === "number") {
    // llama.cpp's OpenAI-compat endpoint reads `repeat_penalty`. Modes still
    // use the HuggingFace name internally; map it so they override request defaults.
    body.repeat_penalty = params.repetition_penalty;
  }
  if (params.chat_template_kwargs && typeof params.chat_template_kwargs === "object") {
    body.chat_template_kwargs = { ...params.chat_template_kwargs };
  }
}
