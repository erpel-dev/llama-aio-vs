import * as path from "path";
import type { ConfigAccessor } from "./config";
import { readEffectiveModelCapabilities, type ModelCapabilities } from "./ggufMetadata";
import { checkLoraAdapter, type LoraFit } from "./loraAdapter";
import { canonicalModelPath, listLoraAdapterEntries, listPleTableEntries } from "./modelLibrary";

/** A file found on disk that could be attached to the selected model. */
export interface AttachmentCandidate {
  path: string;
  sizeBytes: number;
  /** Where it was found, e.g. "Llama AIO" or "Next to the model". */
  source: string;
  /** False when attaching it cannot work; `note` says why. */
  usable: boolean;
  /** Short verdict for the picker row. */
  verdict: string;
  /** One sentence of detail; empty for a clean match. */
  note: string;
  /** Already attached to the model. */
  attached: boolean;
}

const LORA_VERDICT: Record<LoraFit, string> = {
  "match": "Matches this model",
  "unknown-base": "Same model family, base model not named",
  "base-mismatch": "Same model family, different base model",
  "wrong-architecture": "Different model family",
  "invalid": "Not a usable LoRA adapter",
};
const LORA_ORDER: LoraFit[] = ["match", "unknown-base", "base-mismatch", "wrong-architecture", "invalid"];

function samePath(a: string, b: string): boolean {
  return !!a && !!b && canonicalModelPath(a) === canonicalModelPath(b);
}

/**
 * LoRA adapters in the library, best fit first. Only the model family is a
 * hard rule (llama.cpp refuses another one); a base-model name is free text,
 * so a mismatch there stays selectable with its warning.
 */
export function findLoraCandidates(
  config: ConfigAccessor,
  modelPath: string,
  model: ModelCapabilities | undefined,
  attachedPaths: readonly string[] = []
): AttachmentCandidate[] {
  return listLoraAdapterEntries(config, modelPath)
    .map(entry => {
      const check = checkLoraAdapter(entry.path, model);
      return {
        fit: check.fit,
        candidate: {
          path: entry.path,
          sizeBytes: entry.sizeBytes ?? 0,
          source: entry.source,
          usable: check.fit !== "wrong-architecture" && check.fit !== "invalid",
          verdict: LORA_VERDICT[check.fit],
          note: check.message,
          attached: attachedPaths.some(p => samePath(p, entry.path)),
        },
      };
    })
    .sort((a, b) => LORA_ORDER.indexOf(a.fit) - LORA_ORDER.indexOf(b.fit) ||
      path.basename(a.candidate.path).localeCompare(path.basename(b.candidate.path)))
    .map(x => x.candidate);
}

/**
 * PLE-only tables in the library, fitting ones first. A table fits when the
 * same validation that runs on attach accepts it for this model. Matching
 * dimensions do not prove it came from the same checkpoint.
 */
export function findPleCandidates(
  config: ConfigAccessor,
  modelPath: string,
  currentPath = ""
): AttachmentCandidate[] {
  return listPleTableEntries(config, modelPath)
    .map(entry => {
      let note = "";
      try {
        readEffectiveModelCapabilities(modelPath, entry.path);
      } catch (e) {
        note = e instanceof Error ? e.message : String(e);
      }
      return {
        path: entry.path,
        sizeBytes: entry.sizeBytes ?? 0,
        source: entry.source,
        usable: !note,
        verdict: note ? "Does not fit this model" : "Dimensions match this model",
        note,
        attached: samePath(currentPath, entry.path),
      };
    })
    .sort((a, b) => Number(b.usable) - Number(a.usable) || path.basename(a.path).localeCompare(path.basename(b.path)));
}
