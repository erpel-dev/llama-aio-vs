/**
 * Labels and fit badges for the local-model picker (review mockup M3).
 */
import * as os from "os";
import * as path from "path";
import {
  clampLoadSettingsToModel,
  type ModelCapabilities,
  readModelCapabilities,
} from "./ggufMetadata";
import { detectGpus, type GpuMemoryInfo } from "./gpuInfo";
import {
  estimateMemory,
  formatBytes,
  VRAM_HEADROOM_BYTES,
  VRAM_SOFT_HEADROOM_BYTES,
  type MemoryEstimate,
} from "./memoryEstimate";
import {
  displayGgufTitle,
  findSiblingMmproj,
  type LocalModelEntry,
} from "./modelLibrary";
import { DEFAULT_LOAD_SETTINGS, type LlamaLoadSettings } from "./types";

export type ModelFitBadge = "fits" | "tight" | "wont-fit" | "unknown";

export interface ModelPickerHints {
  title: string;
  quant?: string;
  folder: string;
  kind: string;
  contextLabel?: string;
  tools?: boolean;
  vision: boolean;
  shardCount: number;
  fit: ModelFitBadge;
  fitDetail?: string;
}

const QUANT_RE =
  /((?:UD-)?(?:IQ|Q|F|BF)\d+(?:_[A-Z0-9]+)+|MXFP\d+|F16|F32|BF16|Q\d+_K(?:_[A-Z]+)?)$/i;

export function quantFromGgufName(filePath: string): string | undefined {
  const title = displayGgufTitle(filePath);
  const m = QUANT_RE.exec(title);
  return m?.[1];
}

export function shortHomePath(filePath: string): string {
  const home = os.homedir();
  if (home && (filePath === home || filePath.startsWith(home + path.sep))) {
    return "~" + filePath.slice(home.length);
  }
  return filePath;
}

export function describeModelKind(caps?: ModelCapabilities): string {
  if (!caps) {
    return "";
  }
  if (caps.isMoe && caps.expertCount) {
    const used = caps.expertUsedCount ? `×${caps.expertUsedCount}` : "";
    return `MoE ${caps.expertCount}${used}`;
  }
  return "dense";
}

/** Instruct / coder / chat GGUFs almost always speak tools; embedders do not. */
export function modelLikelySupportsTools(filePath: string, caps?: ModelCapabilities): boolean | undefined {
  const name = `${path.basename(filePath)} ${caps?.name || ""} ${caps?.architecture || ""}`.toLowerCase();
  if (/\b(embed|embedding|rerank|clip|mmproj|vocab|whisper|tts)\b/.test(name)) {
    return false;
  }
  if (/\b(instruct|chat|coder|it\b|tool)/.test(name)) {
    return true;
  }
  return undefined;
}

export function classifyModelFit(
  est: MemoryEstimate | undefined,
  ramTotal = os.totalmem()
): { fit: ModelFitBadge; detail?: string } {
  if (!est) {
    return { fit: "unknown" };
  }
  const ram = est.systemRamTotalBytes || ramTotal;
  const vram = est.gpuTotalBytes || 0;
  const gpuNeed = est.totalGpuBytes;
  const cpuNeed = est.totalCpuBytes;
  const cpuLayers = Math.max(0, (est.layersTotal || 0) - (est.layersOnGpu || 0));

  if (vram > 0 && est.willSpill) {
    const room = Math.max(0, vram - VRAM_HEADROOM_BYTES);
    const overflow = Math.max(0, gpuNeed - room);
    if (ram > 0 && cpuNeed + overflow > ram * 0.92) {
      const over = cpuNeed + overflow - ram;
      return {
        fit: "wont-fit",
        detail: over > 0 ? `${formatBytes(over)} over RAM+VRAM` : "won't fit in RAM+VRAM",
      };
    }
    return {
      fit: "tight",
      detail: cpuLayers > 0 ? `needs ${cpuLayers} layers on CPU` : "tight on VRAM",
    };
  }

  if (vram > 0 && gpuNeed > vram - VRAM_SOFT_HEADROOM_BYTES) {
    return { fit: "tight", detail: "tight on VRAM" };
  }
  if (ram > 0 && cpuNeed > ram * 0.85) {
    return { fit: "tight", detail: "tight on system RAM" };
  }
  return { fit: "fits" };
}

export function buildModelPickerHints(
  entry: LocalModelEntry,
  options?: {
    caps?: ModelCapabilities;
    settings?: LlamaLoadSettings;
    gpu?: GpuMemoryInfo;
    gpus?: GpuMemoryInfo[];
    cpuOnly?: boolean;
    ramTotal?: number;
    vision?: boolean;
  }
): ModelPickerHints {
  const caps = options?.caps;
  const title = displayGgufTitle(entry.path);
  const settings = caps
    ? clampLoadSettingsToModel(options?.settings || DEFAULT_LOAD_SETTINGS, caps)
    : options?.settings || DEFAULT_LOAD_SETTINGS;
  const est = caps
    ? estimateMemory(caps, settings, options?.cpuOnly ? undefined : options?.gpu || options?.gpus?.[0], {
        cpuOnly: !!options?.cpuOnly,
        gpus: options?.cpuOnly ? undefined : options?.gpus,
      })
    : undefined;
  const { fit, detail } = classifyModelFit(est, options?.ramTotal);
  const ctx = caps?.maxContextLength;
  return {
    title,
    quant: quantFromGgufName(entry.path),
    folder: shortHomePath(path.dirname(entry.path)),
    kind: describeModelKind(caps),
    contextLabel: ctx && ctx > 0 ? `ctx ${formatContext(ctx)}` : undefined,
    tools: modelLikelySupportsTools(entry.path, caps),
    vision: options?.vision ?? !!findSiblingMmproj(entry.path),
    shardCount: entry.shardCount || 1,
    fit,
    fitDetail: detail,
  };
}

export function formatContext(tokens: number): string {
  if (tokens >= 1024 && tokens % 1024 === 0) {
    return `${tokens / 1024}k`;
  }
  if (tokens >= 1000) {
    return `${Math.round(tokens / 1000)}k`;
  }
  return String(tokens);
}

export function formatPickerDetail(hints: ModelPickerHints): string {
  const bits: string[] = [hints.folder];
  if (hints.kind) {
    bits.push(hints.kind);
  }
  if (hints.contextLabel) {
    bits.push(hints.contextLabel);
  }
  if (hints.tools !== undefined) {
    bits.push(`tools ${hints.tools ? "✓" : "✗"}`);
  }
  bits.push(`vision ${hints.vision ? "✓" : "✗"}`);
  if (hints.fitDetail) {
    bits.push(hints.fitDetail);
  }
  return bits.join(" · ");
}

export function readPickerCapabilities(filePath: string): ModelCapabilities | undefined {
  try {
    return readModelCapabilities(filePath);
  } catch {
    return undefined;
  }
}

/** GPU list for a picker pass (already cached inside detectGpus). */
export function pickerGpus(cpuOnly: boolean, llamaServerBinary?: string): GpuMemoryInfo[] {
  if (cpuOnly) {
    return [];
  }
  return detectGpus(false, llamaServerBinary);
}
