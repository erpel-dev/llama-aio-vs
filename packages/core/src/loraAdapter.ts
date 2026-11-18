import { sameModelFile } from "./processIdentity";
import * as path from "path";
import { effectiveLoraScale, type LoraAdapterSetting } from "./types";
import { readLoraAdapterInfo, type LoraAdapterInfo, type ModelCapabilities } from "./ggufMetadata";

export type LoraFit = "match" | "unknown-base" | "base-mismatch" | "wrong-architecture" | "invalid";

export interface LoraCheck {
  fit: LoraFit;
  /** One sentence for the UI. Empty for a clean match. */
  message: string;
  info?: LoraAdapterInfo;
}

/** Lowercase letters and digits only, so "Qwen3.5-2B" and "qwen3.5_2b" compare equal. */
function nameKey(name: string | undefined): string {
  return (name || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
}

/** Names the model may reasonably be known by: its own name, its base, and its file name without the quant. */
function modelNameKeys(model: ModelCapabilities): string[] {
  const file = path.basename(model.path || "").replace(/\.gguf$/i, "")
    .replace(/[-_.](?:i?q\d[\w]*|f16|f32|bf16)$/i, "");
  return [model.name, model.baseModelName, file].map(nameKey).filter(Boolean);
}

/**
 * What can be known before loading. llama.cpp refuses a different architecture;
 * the base model name is free text, so a mismatch there is only a warning. Tensor
 * shapes are checked by the server at load.
 */
export function checkLoraAdapter(adapterPath: string, model: ModelCapabilities | undefined): LoraCheck {
  let info: LoraAdapterInfo;
  try {
    info = readLoraAdapterInfo(adapterPath);
  } catch (e) {
    return { fit: "invalid", message: e instanceof Error ? e.message : String(e) };
  }
  if (model?.architecture && info.architecture && info.architecture !== model.architecture) {
    return {
      fit: "wrong-architecture", info,
      message: `Made for the ${info.architecture} model family; this model is ${model.architecture}. llama.cpp will refuse it.`,
    };
  }
  if (!info.baseModelName || !model) {
    return { fit: "unknown-base", info, message: "The adapter does not name its base model. It loads only if it was trained on this model." };
  }
  const base = nameKey(info.baseModelName);
  const matches = modelNameKeys(model).some(key => key === base || key.includes(base) || base.includes(key));
  return matches
    ? { fit: "match", info, message: "" }
    : { fit: "base-mismatch", info, message: `Made for ${info.baseModelName}. It loads only if this model has the same base.` };
}

/** llama-server's load error for an adapter, in plain words. Undefined when the line is about something else. */
export function describeLoraLoadError(logLine: string): string | undefined {
  if (/LoRA arch mismatch/i.test(logLine)) {
    return "A LoRA adapter was made for a different model family. Remove it, then start again.";
  }
  if (/maybe wrong base model/i.test(logLine)) {
    return "A LoRA adapter does not fit this model: it was trained on a different base model or size. Remove it, then start again.";
  }
  if (/failed to (?:load|apply) lora adapter|expect general\.type to be 'adapter'|expect adapter\.type to be 'lora'/i.test(logLine)) {
    return "llama-server could not load a LoRA adapter. Remove it, then start again.";
  }
  return undefined;
}

/**
 * Body for POST /lora-adapters, covering every adapter the server loaded.
 *
 * `loaded` is the file list the server was started with, in --lora order; the
 * server reports the same order and the path it was given, so adapters are
 * matched by path with position as a fallback. Each loaded adapter gets the
 * strength the sidebar now wants. One that was removed in the sidebar but is
 * still loaded keeps the strength the server reports until Reload, and one
 * that was added but is not loaded yet is left out: both are pending changes.
 */
export function planLoraScales(
  loaded: string[],
  wanted: LoraAdapterSetting[],
  server: Array<{ id: number; path?: string; scale?: number }>
): Array<{ id: number; scale: number }> {
  return loaded.map((loadedPath, index) => {
    const hit = server.find(s => typeof s.path === "string" && sameModelFile(s.path, loadedPath)) ?? server[index];
    if (!hit) throw new Error(`The server did not load ${path.basename(loadedPath)}.`);
    const want = wanted.find(a => sameModelFile(a.path, loadedPath));
    const current = Number(hit.scale);
    return { id: hit.id, scale: want ? effectiveLoraScale(want) : Number.isFinite(current) ? current : 0 };
  });
}

/**
 * The base model a file names for itself, when that differs from its own name.
 * Quantized uploads usually name the original model as their base, so the same
 * name says nothing. A merged LoRA leaves no trace in a GGUF; this is lineage only.
 */
export function modelLineage(model: ModelCapabilities | undefined): { baseModelName: string; finetune: string } | undefined {
  const base = nameKey(model?.baseModelName);
  if (!model || !base) return undefined;
  const own = nameKey(model.name);
  // "it" / "instruct" mark the ordinary chat release of a base model, not a fine-tune.
  const ordinary = !model.finetune || /^(?:it|instruct|chat|base)$/i.test(model.finetune.trim());
  if (ordinary && own.includes(base)) return undefined;
  return { baseModelName: model.baseModelName!, finetune: model.finetune || "" };
}
