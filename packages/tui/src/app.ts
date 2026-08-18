import path from "node:path";
import {
  createCliRenderer,
  Box,
  BoxRenderable,
  Input,
  InputRenderable,
  InputRenderableEvents,
  Select,
  SelectRenderable,
  SelectRenderableEvents,
  ScrollBox,
  ScrollBoxRenderable,
  Text,
  TextRenderable,
  TabSelect,
  TabSelectRenderable,
  TabSelectRenderableEvents,
  RenderableEvents,
  instantiate,
  type SelectOption,
} from "@opentui/core";
import {
  estimateMemory,
  formatBytes,
  formatGpuDeviceLabel,
  formatLicenseQuickPick,
  formatModelSize,
  languageGgufFiles,
  licenseFromTags,
  listLocalModelEntries,
  preferredMmprojFile,
  preferredMtpDraftFile,
  STARTER_MODEL,
  streamChatCompletion,
  parseTensorSplit,
  tensorSplitForMainShare,
  mainShareFromSplit,
  isLegacyGpu0FirstSplit,
  alignTensorSplitToMainGpu,
  type ChatMessage,
  type HfFileHit,
  type HfModelHit,
  type FlashAttention,
  type KvCacheType,
  type LlamaLoadSettings,
  type MemoryBarChart,
  type ModelLicenseInfo,
  type UiBackend,
  detectGpus,
} from "@llama-aio/core";
import type { AppServices } from "./services.js";
import { theme } from "./theme.js";
import { barCell, barParts, chartSubtitle, memToneFg } from "./memBars.js";
import {
  CPU_THREAD_MAX,
  LOAD_FIELD_DEFS,
  formatFieldValue,
  roundToStep,
} from "./loadForm.js";

type PaneId = "status" | "backend" | "model" | "load" | "chat";

const PANE_TABS: Array<{ name: string; description: string; value: PaneId }> = [
  { name: "Status", description: "Server start / stop / reload", value: "status" },
  { name: "Backend", description: "llama.cpp binary", value: "backend" },
  { name: "Model", description: "Local GGUF library", value: "model" },
  { name: "Load", description: "Context, offload, KV cache", value: "load" },
  { name: "Chat", description: "Talk to the running model", value: "chat" },
];

const PANE_ORDER: PaneId[] = PANE_TABS.map((t) => t.value);

/** Rows needed to show every option (name + optional description line). */
function selectHeight(optionCount: number, showDescription = true): number {
  const linesPer = showDescription ? 2 : 1;
  return Math.max(linesPer, optionCount * linesPer);
}

const SELECTED_MUTED_BG = "#2a3140";
const SELECTED_MUTED_FG = theme.text;

type SelectInternals = {
  scrollOffset: number;
  linesPerItem: number;
};

type TabSelectInternals = {
  scrollOffset: number;
  _tabWidth: number;
  _options: Array<{ name: string }>;
};

function wireSelectFocusStyle(sel: SelectRenderable): void {
  const paint = (focused: boolean) => {
    sel.selectedBackgroundColor = focused ? theme.selectedBg : SELECTED_MUTED_BG;
    sel.selectedTextColor = focused ? theme.selectedFg : SELECTED_MUTED_FG;
  };
  sel.on(RenderableEvents.FOCUSED, () => paint(true));
  sel.on(RenderableEvents.BLURRED, () => paint(false));
  paint(false);
}

/** Click to highlight (and optionally activate); wheel to scroll the list. */
function wireSelectMouse(sel: SelectRenderable, activateOnClick: boolean): void {
  sel.onMouseDown = (event) => {
    if (event.button !== 0) {
      return;
    }
    event.stopPropagation();
    sel.focus();
    const { scrollOffset, linesPerItem } = sel as unknown as SelectInternals;
    const localY = event.y - sel.y;
    if (localY < 0 || localY >= sel.height) {
      return;
    }
    const row = Math.floor(localY / Math.max(1, linesPerItem || 1));
    const index = (scrollOffset || 0) + row;
    if (index < 0 || index >= sel.options.length) {
      return;
    }
    if (sel.getSelectedIndex() !== index) {
      sel.setSelectedIndex(index);
    }
    if (activateOnClick) {
      sel.selectCurrent();
    }
  };
  sel.onMouseScroll = (event) => {
    const dir = event.scroll?.direction;
    const steps = Math.max(1, event.scroll?.delta ?? 1);
    if (dir === "up") {
      sel.moveUp(steps);
      event.stopPropagation();
    } else if (dir === "down") {
      sel.moveDown(steps);
      event.stopPropagation();
    }
  };
}

function wireSelect(sel: SelectRenderable, activateOnClick: boolean): void {
  wireSelectFocusStyle(sel);
  wireSelectMouse(sel, activateOnClick);
}

function wireTabSelectMouse(tabs: TabSelectRenderable): void {
  tabs.onMouseDown = (event) => {
    if (event.button !== 0) {
      return;
    }
    event.stopPropagation();
    tabs.focus();
    const internal = tabs as unknown as TabSelectInternals;
    const tabWidth = Math.max(1, internal._tabWidth || 12);
    const localX = event.x - tabs.x;
    if (localX < 0 || localX >= tabs.width) {
      return;
    }
    const index = (internal.scrollOffset || 0) + Math.floor(localX / tabWidth);
    const count = internal._options?.length ?? 0;
    if (index < 0 || index >= count) {
      return;
    }
    if (tabs.getSelectedIndex() !== index) {
      tabs.setSelectedIndex(index);
    }
  };
}
const LOAD_PRESETS: Record<
  string,
  { label: string; contextLength: number | "fit"; cacheTypeK: KvCacheType; cacheTypeV: KvCacheType }
> = {
  agent: {
    label: "Coding agent",
    contextLength: 65536,
    cacheTypeK: "q8_0",
    cacheTypeV: "q8_0",
  },
  context: {
    label: "Max context",
    contextLength: "fit",
    cacheTypeK: "q8_0",
    cacheTypeV: "q4_0",
  },
  quality: {
    label: "Max quality",
    contextLength: 65536,
    cacheTypeK: "f16",
    cacheTypeV: "q8_0",
  },
};

const FIT_CONTEXT_STEPS = [
  262144, 196608, 163840, 131072, 98304, 65536, 49152, 32768, 24576, 16384, 8192,
];

function shortPath(p: string, max = 56): string {
  if (!p) {
    return "(none)";
  }
  if (p.length <= max) {
    return p;
  }
  return `…${p.slice(-(max - 1))}`;
}

function statusLabel(running: boolean, starting: boolean, dirty: boolean): string {
  if (starting && !running) {
    return "starting…";
  }
  if (running && dirty) {
    return "running · settings changed — F12 reload";
  }
  if (running) {
    return "running";
  }
  if (dirty) {
    return "stopped · settings changed — F12 reload";
  }
  return "stopped";
}

function statusColor(running: boolean, starting: boolean, dirty: boolean): string {
  if (starting && !running) {
    return theme.accent;
  }
  if (dirty) {
    return theme.warn;
  }
  if (running) {
    return theme.ok;
  }
  return theme.bad;
}

export async function runApp(services: AppServices): Promise<void> {
  const renderer = await createCliRenderer({
    exitOnCtrlC: true,
    backgroundColor: theme.bg,
    useMouse: true,
    autoFocus: true,
  });

  // OpenTUI factory helpers return VNodes; mutations/events only stick after
  // instantiate(). Keep live Renderable references for everything we update.
  const mount = (node: unknown) => instantiate(renderer, node as never);

  let activePane: PaneId = "status";
  let busy = false;
  let bootMessage = "";
  let statusMessage = "";
  let chatBusy = false;
  let chatAbort: AbortController | undefined;
  const chatHistory: ChatMessage[] = [];
  let chatDraft = "";

  // ─── shell ─────────────────────────────────────────────────────────────
  const headerStatus = mount(
    Text({
      content: "",
      fg: theme.muted,
      flexGrow: 1,
    })
  ) as TextRenderable;

  const header = mount(
    Box(
      {
        width: "100%",
        height: 3,
        border: true,
        borderStyle: "rounded",
        borderColor: theme.border,
        backgroundColor: theme.panel,
        paddingLeft: 1,
        paddingRight: 1,
        flexDirection: "row",
        alignItems: "center",
      },
      Text({ content: "Llama AIO", fg: theme.accent, attributes: 1 }),
      Text({ content: "  ", fg: theme.muted }),
      headerStatus
    )
  ) as BoxRenderable;

  const tabs = mount(
    TabSelect({
      width: "100%",
      height: 2,
      options: PANE_TABS,
      tabWidth: 12,
      showDescription: false,
      showUnderline: true,
      backgroundColor: theme.bg,
      textColor: theme.muted,
      selectedBackgroundColor: theme.selectedBg,
      selectedTextColor: theme.selectedFg,
      focusedBackgroundColor: theme.panel,
    })
  ) as TabSelectRenderable;
  wireTabSelectMouse(tabs);

  const body = mount(
    Box({
      width: "100%",
      flexGrow: 1,
      flexDirection: "column",
      backgroundColor: theme.bg,
      padding: 0,
    })
  ) as BoxRenderable;

  const footer = mount(
    Text({
      content:
        "Tab/F1-F5 · Load: ↑↓ Enter ←→ Esc · F12 reload · s/x/r · q",
      fg: theme.muted,
      height: 1,
    })
  ) as TextRenderable;

  const root = mount(
    Box({
      width: "100%",
      height: "100%",
      flexDirection: "column",
      backgroundColor: theme.bg,
      padding: 1,
      gap: 1,
    })
  ) as BoxRenderable;
  root.add(header);
  root.add(tabs);
  root.add(body);
  root.add(footer);
  renderer.root.add(root);

  // ─── Status pane ───────────────────────────────────────────────────────
  const statusInfo = mount(
    Text({
      content: "",
      fg: theme.text,
      flexGrow: 1,
      minHeight: 6,
      wrapMode: "word",
    })
  ) as TextRenderable;

  const statusActionsOpts = [
    { name: "Start server", description: "Launch llama-server with current model/settings", value: "start" },
    { name: "Stop server", description: "Stop the shared llama-server", value: "stop" },
    { name: "Reload server", description: "Apply dirty load settings", value: "reload" },
    { name: "Refresh", description: "Re-read status and config", value: "refresh" },
  ];
  const statusActions = mount(
    Select({
      width: "100%",
      height: selectHeight(statusActionsOpts.length),
      showDescription: true,
      showScrollIndicator: false,
      backgroundColor: theme.panel,
      focusedBackgroundColor: theme.panel,
      selectedBackgroundColor: theme.selectedBg,
      selectedTextColor: theme.selectedFg,
      textColor: theme.text,
      descriptionColor: theme.muted,
      options: statusActionsOpts,
    })
  ) as SelectRenderable;
  wireSelect(statusActions, true);

  const statusPane = mount(
    Box({
      width: "100%",
      height: "100%",
      flexDirection: "column",
      gap: 1,
      border: true,
      borderStyle: "rounded",
      borderColor: theme.border,
      backgroundColor: theme.panel,
      padding: 1,
      title: " Server ",
      titleColor: theme.accent,
    })
  ) as BoxRenderable;
  statusPane.add(statusInfo);
  statusPane.add(statusActions);

  // ─── Backend pane ──────────────────────────────────────────────────────
  const backendHint = mount(
    Text({
      content: "",
      fg: theme.muted,
      height: 2,
      wrapMode: "word",
    })
  ) as TextRenderable;

  const backendSelect = mount(
    Select({
      width: "100%",
      flexGrow: 1,
      showDescription: true,
      showScrollIndicator: true,
      backgroundColor: theme.panel,
      focusedBackgroundColor: theme.panel,
      selectedBackgroundColor: theme.selectedBg,
      selectedTextColor: theme.selectedFg,
      textColor: theme.text,
      descriptionColor: theme.muted,
      options: [],
    })
  ) as SelectRenderable;
  wireSelect(backendSelect, false);

  const backendActionsOpts = [
    { name: "Use selected backend", description: "Write backend to shared config", value: "use" },
    { name: "Install / upgrade", description: "Download latest release for selected backend", value: "install" },
    { name: "Refresh list", description: "Re-scan installs and PATH", value: "refresh" },
  ];
  const backendActions = mount(
    Select({
      width: "100%",
      height: selectHeight(backendActionsOpts.length),
      showDescription: true,
      showScrollIndicator: false,
      backgroundColor: theme.panel,
      focusedBackgroundColor: theme.panel,
      selectedBackgroundColor: theme.selectedBg,
      selectedTextColor: theme.selectedFg,
      textColor: theme.text,
      descriptionColor: theme.muted,
      options: backendActionsOpts,
    })
  ) as SelectRenderable;
  wireSelect(backendActions, true);

  const backendPane = mount(
    Box({
      width: "100%",
      height: "100%",
      flexDirection: "column",
      gap: 1,
      border: true,
      borderStyle: "rounded",
      borderColor: theme.border,
      backgroundColor: theme.panel,
      padding: 1,
      title: " Backend ",
      titleColor: theme.accent,
    })
  ) as BoxRenderable;
  backendPane.add(backendHint);
  backendPane.add(backendSelect);
  backendPane.add(backendActions);

  // ─── Model pane ────────────────────────────────────────────────────────
  type ModelBrowseMode = "local" | "hf-query" | "hf-repos" | "hf-confirm" | "hf-files";
  let modelBrowse: ModelBrowseMode = "local";
  let hfRepos: HfModelHit[] = [];
  let hfLicenses = new Map<string, ModelLicenseInfo>();
  let hfPickedRepo: HfModelHit | undefined;
  let hfPickedLicense: ModelLicenseInfo | undefined;
  let hfFiles: HfFileHit[] = [];
  let hfAllFiles: HfFileHit[] = [];

  const modelHint = mount(
    Text({
      content: "",
      fg: theme.muted,
      height: 2,
      wrapMode: "word",
    })
  ) as TextRenderable;

  const modelSearch = mount(
    Input({
      width: "100%",
      visible: false,
      placeholder: "Hugging Face search · Enter to search · Esc cancel",
      backgroundColor: theme.inputBg,
      focusedBackgroundColor: theme.inputBg,
      textColor: theme.text,
      placeholderColor: theme.muted,
    })
  ) as InputRenderable;

  const modelSelect = mount(
    Select({
      width: "100%",
      flexGrow: 1,
      showDescription: true,
      showScrollIndicator: true,
      backgroundColor: theme.panel,
      focusedBackgroundColor: theme.panel,
      selectedBackgroundColor: theme.selectedBg,
      selectedTextColor: theme.selectedFg,
      textColor: theme.text,
      descriptionColor: theme.muted,
      options: [],
    })
  ) as SelectRenderable;
  wireSelect(modelSelect, true);

  const modelPane = mount(
    Box({
      width: "100%",
      height: "100%",
      flexDirection: "column",
      gap: 1,
      border: true,
      borderStyle: "rounded",
      borderColor: theme.border,
      backgroundColor: theme.panel,
      padding: 1,
      title: " Model ",
      titleColor: theme.accent,
    })
  ) as BoxRenderable;
  modelPane.add(modelHint);
  modelPane.add(modelSearch);
  modelPane.add(modelSelect);

  // ─── Load pane ─────────────────────────────────────────────────────────
  const MEM_BAR_WIDTH = 48;

  function memTitleRow(title: string): {
    row: BoxRenderable;
    title: TextRenderable;
    sub: TextRenderable;
  } {
    const titleText = mount(
      Text({ content: title, fg: theme.text, attributes: 1, flexGrow: 1 })
    ) as TextRenderable;
    const subText = mount(Text({ content: "—", fg: theme.muted })) as TextRenderable;
    const row = mount(
      Box({
        width: "100%",
        height: 1,
        flexDirection: "row",
        flexShrink: 0,
        gap: 1,
      })
    ) as BoxRenderable;
    row.add(titleText);
    row.add(subText);
    return { row, title: titleText, sub: subText };
  }

  function memBarRow(): {
    row: BoxRenderable;
    weights: TextRenderable;
    vision: TextRenderable;
    draft: TextRenderable;
    kv: TextRenderable;
    overhead: TextRenderable;
    free: TextRenderable;
  } {
    const mk = (fg: string) =>
      mount(Text({ content: "", fg, height: 1, flexShrink: 0 })) as TextRenderable;
    const weights = mk(theme.memWeights);
    const vision = mk(theme.memVision);
    const draft = mk(theme.memDraft);
    const kv = mk(theme.memKv);
    const overhead = mk(theme.memOverhead);
    const free = mk(theme.memTrack);
    const row = mount(
      Box({
        width: "100%",
        height: 1,
        flexDirection: "row",
        flexShrink: 0,
      })
    ) as BoxRenderable;
    row.add(weights);
    row.add(vision);
    row.add(draft);
    row.add(kv);
    row.add(overhead);
    row.add(free);
    return { row, weights, vision, draft, kv, overhead, free };
  }

  function paintMemBar(
    cells: ReturnType<typeof memBarRow>,
    chart: MemoryBarChart | null,
    width: number
  ): void {
    const empty = { weights: "", vision: "", draft: "", kv: "", overhead: "", free: "░".repeat(width) };
    if (!chart) {
      cells.weights.content = empty.weights;
      cells.vision.content = empty.vision;
      cells.draft.content = empty.draft;
      cells.kv.content = empty.kv;
      cells.overhead.content = empty.overhead;
      cells.free.content = empty.free;
      cells.free.fg = theme.memTrack as never;
      return;
    }
    const parts = barParts(chart, width);
    const byKey = {
      weights: "",
      vision: "",
      draft: "",
      kv: "",
      overhead: "",
      free: "",
    };
    for (const p of parts) {
      byKey[p.key] = barCell(p.cols);
    }
    cells.weights.content = byKey.weights;
    cells.vision.content = byKey.vision;
    cells.draft.content = byKey.draft;
    cells.kv.content = byKey.kv;
    cells.overhead.content = byKey.overhead;
    cells.free.content =
      byKey.free ||
      (byKey.weights || byKey.vision || byKey.draft || byKey.kv || byKey.overhead ? "" : "░".repeat(width));
  }

  const loadMemNote = mount(
    Text({
      content: "Bars = estimate at full context. Live GPU free is occupancy, not the bar.",
      fg: theme.muted,
      wrapMode: "word",
      flexShrink: 0,
    })
  ) as TextRenderable;

  const loadVramHead = memTitleRow("VRAM · est. at full context");
  const loadVramBar = memBarRow();
  const loadVram2Head = memTitleRow("VRAM · GPU 1 · est. at full context");
  const loadVram2Bar = memBarRow();
  const loadRamHead = memTitleRow("System RAM · est. at full context");
  const loadRamBar = memBarRow();

  const loadMemLegendRow = mount(
    Box({
      width: "100%",
      height: 1,
      flexDirection: "row",
      gap: 2,
      flexShrink: 0,
    })
  ) as BoxRenderable;
  loadMemLegendRow.add(
    mount(Text({ content: "█ Weights", fg: theme.memWeights })) as TextRenderable
  );
  loadMemLegendRow.add(
    mount(Text({ content: "█ Vision", fg: theme.memVision })) as TextRenderable
  );
  loadMemLegendRow.add(
    mount(Text({ content: "█ Spec", fg: theme.memDraft })) as TextRenderable
  );
  loadMemLegendRow.add(mount(Text({ content: "█ KV cache", fg: theme.memKv })) as TextRenderable);
  loadMemLegendRow.add(
    mount(Text({ content: "█ Overhead", fg: theme.memOverhead })) as TextRenderable
  );

  const loadMemSummary = mount(
    Text({
      content: "Select a model to estimate VRAM / RAM use.",
      fg: theme.muted,
      wrapMode: "word",
      flexShrink: 0,
    })
  ) as TextRenderable;

  const loadMemWarn = mount(
    Text({
      content: "",
      fg: theme.warn,
      wrapMode: "word",
      flexShrink: 0,
    })
  ) as TextRenderable;

  const loadHint = mount(
    Text({
      content: "",
      fg: theme.muted,
      height: 2,
      wrapMode: "word",
      flexShrink: 0,
    })
  ) as TextRenderable;

  const loadMemPanel = mount(
    Box({
      width: "100%",
      flexDirection: "column",
      gap: 0,
      flexShrink: 0,
      border: true,
      borderStyle: "rounded",
      borderColor: theme.border,
      padding: 1,
      title: " Memory estimate ",
      titleColor: theme.accent,
    })
  ) as BoxRenderable;
  loadMemPanel.add(loadMemNote);
  loadMemPanel.add(loadVramHead.row);
  loadMemPanel.add(loadVramBar.row);
  loadMemPanel.add(loadVram2Head.row);
  loadMemPanel.add(loadVram2Bar.row);
  loadMemPanel.add(loadRamHead.row);
  loadMemPanel.add(loadRamBar.row);
  loadMemPanel.add(loadMemLegendRow);
  loadMemPanel.add(loadMemSummary);
  loadMemPanel.add(loadMemWarn);

  // Field list navigation: ↑↓ select · Enter edit/activate · ←→ adjust · Esc done
  let loadEditing = false;
  let loadSaveTimer: ReturnType<typeof setTimeout> | undefined;

  type LoadBounds = { min: number; max: number };

  function splitGpuInfos() {
    const cpuOnly = services.installer.resolveActiveUiBackend() === "cpu";
    return cpuOnly ? [] : detectGpus(false, services.processManager.resolveBinary());
  }

  function splitGpuCount(): number {
    return Math.max(2, splitGpuInfos().length || 2);
  }

  function syncMainGpuOptions(): void {
    const field = LOAD_FIELD_DEFS.find((f) => f.id === "mainGpu");
    if (!field || field.kind !== "enum") {
      return;
    }
    const gpus = splitGpuInfos();
    field.options = gpus.length
      ? gpus.map((g, i) => ({
          value: String(i),
          name: `${formatGpuDeviceLabel(g, i)} · ${formatBytes(g.totalBytes)}`,
        }))
      : Array.from({ length: 8 }, (_, i) => ({ value: String(i), name: `GPU ${i}` }));
  }

  function tensorSplitPercent(load: LlamaLoadSettings): number {
    const gpus = splitGpuInfos();
    const n = splitGpuCount();
    const vram = gpus.length ? gpus.map((g) => g.totalBytes) : Array.from({ length: n }, () => 1);
    const share = mainShareFromSplit(load.tensorSplit, load.mainGpu, n, vram);
    return Math.min(90, Math.max(10, Math.round(share * 100)));
  }

  function visibleLoadFields(): typeof LOAD_FIELD_DEFS {
    const splitMode = services.store.getState().loadSettings.splitMode || "layer";
    return LOAD_FIELD_DEFS.filter((f) => !(f.id === "tensorSplit" && splitMode === "none"));
  }

  function maybeMigrateLegacySplit(): void {
    const gpus = splitGpuInfos();
    if (gpus.length < 2) {
      return;
    }
    const load = services.store.getState().loadSettings;
    if (!isLegacyGpu0FirstSplit(load.tensorSplit) || load.mainGpu <= 0) {
      return;
    }
    const aligned = alignTensorSplitToMainGpu(
      load.tensorSplit,
      load.mainGpu,
      gpus.length,
      gpus.map((g) => g.totalBytes)
    );
    if (!aligned) {
      return;
    }
    void services.store.updateLoadSettings({ tensorSplit: aligned });
  }

  function loadBounds(field: (typeof LOAD_FIELD_DEFS)[number]): LoadBounds {
    if (field.kind !== "number") {
      return { min: 0, max: 0 };
    }
    const caps = services.store.getState().modelCapabilities;
    const min = field.min === "ctxMin" ? 512 : field.min;
    let max: number;
    if (field.max === "ctxMax") {
      max = Math.max(512, caps?.maxContextLength || 131072);
    } else if (field.max === "gpuMax") {
      max = Math.max(1, caps?.blockCount || 128);
    } else if (field.max === "cpuMax") {
      max = CPU_THREAD_MAX;
    } else {
      max = field.max;
    }
    return { min, max: Math.max(min, max) };
  }

  function readFieldRaw(field: (typeof LOAD_FIELD_DEFS)[number]): string | number {
    const state = services.store.getState();
    if (field.kind === "preset") {
      const p = LOAD_PRESETS[field.presetId];
      if (!p) return "";
      return p.contextLength === "fit"
        ? `fit · KV ${p.cacheTypeK}/${p.cacheTypeV}`
        : `${p.contextLength} · KV ${p.cacheTypeK}/${p.cacheTypeV}`;
    }
    if (field.kind === "action") {
      return services.processManager.getStatus().configDirty ? "settings dirty" : "up to date";
    }
    if (field.kind === "enum") {
      const load = state.loadSettings;
      if (field.key === "offloadKvCacheToGpu") {
        return load.offloadKvCacheToGpu ? "true" : "false";
      }
      if (field.key === "mmprojOffloadToGpu") {
        return load.mmprojOffloadToGpu !== false ? "true" : "false";
      }
      if (field.key === "cacheTypeK") {
        return load.cacheTypeK;
      }
      if (field.key === "cacheTypeV") {
        return load.cacheTypeV;
      }
      if (field.key === "flashAttention") {
        return load.flashAttention;
      }
      if (field.key === "speculativeMode") {
        return load.speculativeMode || "off";
      }
      if (field.key === "splitMode") {
        return load.splitMode || "layer";
      }
      if (field.key === "mainGpu") {
        return String(load.mainGpu);
      }
      return "";
    }
    if (field.store === "request") {
      const req = state.requestSettings;
      if (field.key === "temperature") return req.temperature;
      if (field.key === "topP") return req.topP;
      if (field.key === "topK") return req.topK;
      if (field.key === "maxTokens") return req.maxTokens;
      return 0;
    }
    const load = state.loadSettings;
    if (field.key === "contextLength") return load.contextLength;
    if (field.key === "gpuOffload") return load.gpuOffload;
    if (field.key === "cpuThreads") return load.cpuThreads;
    if (field.key === "maxConcurrentPredictions") return load.maxConcurrentPredictions;
    if (field.key === "nCpuMoe") return load.nCpuMoe;
    if (field.key === "evalBatchSize") return load.evalBatchSize;
    if (field.key === "physicalBatchSize") return load.physicalBatchSize;
    if (field.key === "maxDraftTokens") return load.maxDraftTokens;
    if (field.key === "draftGpuOffload") return load.draftGpuOffload;
    if (field.key === "tensorSplit") return tensorSplitPercent(load);
    if (field.key === "mainGpu") return load.mainGpu;
    return 0;
  }

  function formatLoadOption(field: (typeof LOAD_FIELD_DEFS)[number]): { name: string; description: string; value: string } {
    const raw = readFieldRaw(field);
    if (field.kind === "number") {
      const shown = formatFieldValue(raw as number, field.step);
      const suffix = field.key === "tensorSplit" ? "%" : "";
      return {
        name: `${field.label}  ·  ${shown}${suffix}`,
        description: field.help,
        value: field.id,
      };
    }
    if (field.kind === "enum") {
      const cur = String(raw);
      const opt = field.options.find((o) => o.value === cur);
      const shown = opt?.name || cur;
      return {
        name: `${field.label}  ·  ${shown}`,
        description: opt?.name || field.help,
        value: field.id,
      };
    }
    if (field.kind === "preset") {
      return {
        name: field.label,
        description: `${raw} — ${field.help}`,
        value: field.id,
      };
    }
    return {
      name: field.label,
      description: `${raw} — ${field.help}`,
      value: field.id,
    };
  }

  function currentLoadField(): (typeof LOAD_FIELD_DEFS)[number] | undefined {
    const id = loadFieldSelect.getSelectedOption()?.value as string | undefined;
    return LOAD_FIELD_DEFS.find((f) => f.id === id);
  }

  function paintLoadDetailBar(min: number, max: number, value: number): void {
    const width = Math.max(24, Math.min(56, (renderer.width || 80) - 8));
    const range = max - min;
    const ratio = range <= 0 ? 0 : (value - min) / range;
    const fillCols = Math.max(0, Math.min(width, Math.round(ratio * width)));
    const emptyCols = Math.max(0, width - fillCols);
    loadDetailFilled.content = fillCols > 0 ? "█".repeat(fillCols) : "";
    loadDetailEmpty.content = emptyCols > 0 ? "─".repeat(emptyCols) : "";
  }

  function refreshLoadDetail(): void {
    const field = currentLoadField();
    if (!field) {
      loadDetailHelp.content = "";
      loadDetailFilled.content = "";
      loadDetailEmpty.content = "";
      loadEditHint.content = "↑↓ settings · Enter edit/apply · F12 reload dirty";
      loadEditHint.fg = theme.muted as never;
      return;
    }
    loadDetailHelp.content = field.help;
    if (field.kind === "number") {
      const { min, max } = loadBounds(field);
      const value = Number(readFieldRaw(field));
      paintLoadDetailBar(min, max, value);
    } else if (field.kind === "enum") {
      const cur = String(readFieldRaw(field));
      const labels = field.options.map((o) => {
        const label = field.key === "mainGpu" ? o.name : o.value;
        return o.value === cur ? `[${label}]` : label;
      });
      loadDetailFilled.content = "";
      loadDetailEmpty.content = labels.join("  ");
    } else {
      loadDetailFilled.content = "";
      loadDetailEmpty.content = "";
    }
    if (loadEditing && (field.kind === "number" || field.kind === "enum")) {
      loadEditHint.content = "Editing — ←→ change · Enter/Esc done · F12 reload";
      loadEditHint.fg = theme.warn as never;
      loadFieldSelect.selectedBackgroundColor = theme.warn;
    } else {
      loadEditHint.content =
        field.kind === "preset" || field.kind === "action"
          ? "Enter to run · ↑↓ move · F12 reload dirty settings"
          : "Enter to edit · ↑↓ move · ←→ after Enter · F12 reload";
      loadEditHint.fg = theme.muted as never;
      if (!loadFieldSelect.focused) {
        loadFieldSelect.selectedBackgroundColor = SELECTED_MUTED_BG;
      } else {
        loadFieldSelect.selectedBackgroundColor = theme.selectedBg;
      }
    }
  }

  function setLoadEditing(on: boolean): void {
    loadEditing = on;
    if (on) {
      // Blur so OpenTUI Select doesn't eat ↑↓ while we adjust the value.
      if (loadFieldSelect.focused) {
        loadFieldSelect.blur();
      }
    } else if (activePane === "load" && !loadFieldSelect.focused) {
      loadFieldSelect.focus();
    }
    refreshLoadDetail();
    renderer.requestRender();
  }

  function scheduleLoadPatch(apply: () => Promise<void>): void {
    if (loadSaveTimer) {
      clearTimeout(loadSaveTimer);
    }
    loadSaveTimer = setTimeout(() => {
      loadSaveTimer = undefined;
      void apply().catch((err) => {
        statusMessage = err instanceof Error ? err.message : String(err);
        refreshStatus();
        renderer.requestRender();
      });
    }, 120);
  }

  async function commitLoadNumber(field: Extract<(typeof LOAD_FIELD_DEFS)[number], { kind: "number" }>, value: number): Promise<void> {
    if (field.store === "request") {
      await services.store.updateRequestSettings({ [field.key]: value });
      return;
    }
    if (field.key === "tensorSplit") {
      const load = services.store.getState().loadSettings;
      if (load.splitMode === "none") {
        return;
      }
      await services.store.updateLoadSettings({
        tensorSplit: tensorSplitForMainShare(value / 100, load.mainGpu, splitGpuCount()),
      });
      return;
    }
    await services.store.updateLoadSettings({ [field.key]: value });
  }

  async function commitLoadEnum(field: Extract<(typeof LOAD_FIELD_DEFS)[number], { kind: "enum" }>, value: string): Promise<void> {
    if (field.key === "offloadKvCacheToGpu") {
      await services.store.updateLoadSettings({ offloadKvCacheToGpu: value === "true" });
      return;
    }
    if (field.key === "mmprojOffloadToGpu") {
      await services.store.updateLoadSettings({ mmprojOffloadToGpu: value === "true" });
      return;
    }
    if (field.key === "cacheTypeK" || field.key === "cacheTypeV") {
      await services.store.updateLoadSettings({ [field.key]: value as KvCacheType });
      return;
    }
    if (field.key === "flashAttention") {
      await services.store.updateLoadSettings({ flashAttention: value as FlashAttention });
      return;
    }
    if (field.key === "speculativeMode") {
      const mode = value === "mtp" || value === "dflash" ? value : "off";
      await services.store.updateLoadSettings({
        speculativeMode: mode,
        ...(mode === "dflash" && services.store.getState().loadSettings.maxDraftTokens <= 2
          ? { maxDraftTokens: 15 }
          : {}),
      });
      return;
    }
    if (field.key === "mainGpu") {
      const mainGpu = Number(value);
      const load = services.store.getState().loadSettings;
      const patch: Partial<LlamaLoadSettings> = { mainGpu };
      if (parseTensorSplit(load.tensorSplit).length >= 2) {
        patch.tensorSplit = tensorSplitForMainShare(
          tensorSplitPercent(load) / 100,
          mainGpu,
          splitGpuCount()
        );
      }
      await services.store.updateLoadSettings(patch);
      return;
    }
    await services.store.updateLoadSettings({ [field.key]: value });
  }

  function adjustLoadField(dir: -1 | 1): boolean {
    const field = currentLoadField();
    if (!field || (field.kind !== "number" && field.kind !== "enum")) {
      return false;
    }
    if (field.kind === "number") {
      const { min, max } = loadBounds(field);
      const cur = Number(readFieldRaw(field));
      const next = roundToStep(cur + dir * field.step, field.step, min, max);
      if (next === cur) {
        refreshLoadDetail();
        return true;
      }
      // Optimistic UI while debounce writes.
      paintLoadDetailBar(min, max, next);
      const opts = loadFieldSelect.options.map((o) =>
        o.value === field.id
          ? {
              ...o,
              name: `${field.label}  ·  ${formatFieldValue(next, field.step)}${field.key === "tensorSplit" ? "%" : ""}`,
            }
          : o
      );
      loadFieldSelect.options = opts;
      scheduleLoadPatch(() => commitLoadNumber(field, next));
      refreshLoadDetail();
      renderer.requestRender();
      return true;
    }
    const cur = String(readFieldRaw(field));
    const idx = field.options.findIndex((o) => o.value === cur);
    const nextIdx = idx < 0 ? 0 : (idx + dir + field.options.length) % field.options.length;
    const next = field.options[nextIdx]!.value;
    if (next === cur) {
      return true;
    }
    scheduleLoadPatch(() => commitLoadEnum(field, next));
    // refresh via store onDidChange; optimistic option text:
    const nextName = field.options.find((o) => o.value === next)?.name || next;
    loadFieldSelect.options = visibleLoadFields().map((f) =>
      f.id === field.id
        ? {
            name: `${field.label}  ·  ${nextName}`,
            description: field.options.find((o) => o.value === next)?.name || field.help,
            value: field.id,
          }
        : formatLoadOption(f)
    );
    refreshLoadDetail();
    renderer.requestRender();
    return true;
  }

  function activateLoadField(): void {
    const field = currentLoadField();
    if (!field) {
      return;
    }
    if (field.kind === "preset") {
      setLoadEditing(false);
      void applyPreset(field.presetId);
      return;
    }
    if (field.kind === "action") {
      setLoadEditing(false);
      void reloadServer();
      return;
    }
    setLoadEditing(!loadEditing);
  }

  syncMainGpuOptions();
  const loadFieldSelect = mount(
    Select({
      width: "100%",
      flexGrow: 1,
      showDescription: true,
      showScrollIndicator: true,
      backgroundColor: theme.panel,
      focusedBackgroundColor: theme.panel,
      selectedBackgroundColor: theme.selectedBg,
      selectedTextColor: theme.selectedFg,
      textColor: theme.text,
      descriptionColor: theme.muted,
      options: visibleLoadFields().map((f) => formatLoadOption(f)),
    })
  ) as SelectRenderable;
  wireSelect(loadFieldSelect, true);

  const loadDetailHelp = mount(
    Text({
      content: "",
      fg: theme.muted,
      wrapMode: "word",
      flexShrink: 0,
    })
  ) as TextRenderable;

  const loadDetailFilled = mount(
    Text({ content: "", fg: theme.accent, height: 1, flexShrink: 0 })
  ) as TextRenderable;
  const loadDetailEmpty = mount(
    Text({ content: "", fg: theme.border, height: 1, flexShrink: 0 })
  ) as TextRenderable;
  const loadDetailBar = mount(
    Box({
      width: "100%",
      height: 1,
      flexDirection: "row",
      flexShrink: 0,
    })
  ) as BoxRenderable;
  loadDetailBar.add(loadDetailFilled);
  loadDetailBar.add(loadDetailEmpty);

  const loadEditHint = mount(
    Text({
      content: "↑↓ settings · Enter edit/apply · F12 reload dirty",
      fg: theme.muted,
      height: 1,
      flexShrink: 0,
    })
  ) as TextRenderable;

  // Detail bar click/drag adjusts the current number field (enters edit implicitly).
  function loadDetailSetFromX(absX: number): void {
    const field = currentLoadField();
    if (!field || field.kind !== "number") {
      return;
    }
    if (!loadEditing) {
      loadEditing = true;
    }
    const { min, max } = loadBounds(field);
    const width = Math.max(1, loadDetailBar.width || 40);
    const local = Math.max(0, Math.min(width, absX - loadDetailBar.x));
    const raw = min + (local / width) * (max - min);
    const next = roundToStep(raw, field.step, min, max);
    paintLoadDetailBar(min, max, next);
    loadFieldSelect.options = visibleLoadFields().map((f) =>
      f.id === field.id
        ? {
            name: `${field.label}  ·  ${formatFieldValue(next, field.step)}${field.key === "tensorSplit" ? "%" : ""}`,
            description: field.help,
            value: field.id,
          }
        : formatLoadOption(f)
    );
    scheduleLoadPatch(() => commitLoadNumber(field, next));
    refreshLoadDetail();
    renderer.requestRender();
  }
  loadDetailBar.onMouseDown = (event) => {
    if (event.button !== 0) return;
    event.stopPropagation();
    loadDetailSetFromX(event.x);
  };
  loadDetailBar.onMouseDrag = (event) => {
    if (event.button !== 0) return;
    event.stopPropagation();
    loadDetailSetFromX(event.x);
  };

  const loadPane = mount(
    Box({
      width: "100%",
      height: "100%",
      flexDirection: "column",
      gap: 1,
      border: true,
      borderStyle: "rounded",
      borderColor: theme.border,
      backgroundColor: theme.panel,
      padding: 1,
      title: " Load settings ",
      titleColor: theme.accent,
    })
  ) as BoxRenderable;
  loadPane.add(loadMemPanel);
  loadPane.add(loadHint);
  loadPane.add(loadFieldSelect);
  loadPane.add(loadDetailHelp);
  loadPane.add(loadDetailBar);
  loadPane.add(loadEditHint);

  // ─── Chat pane ─────────────────────────────────────────────────────────
  const chatLog = mount(
    Text({
      content: "Select a model, start the server, then type below.",
      fg: theme.text,
      wrapMode: "word",
    })
  ) as TextRenderable;

  const chatScroll = mount(
    ScrollBox({
      width: "100%",
      flexGrow: 1,
      backgroundColor: theme.inputBg,
      stickyScroll: true,
      stickyStart: "bottom",
    })
  ) as ScrollBoxRenderable;
  chatScroll.add(chatLog);

  const chatInput = mount(
    Input({
      width: "100%",
      placeholder: "Message · Enter send · Esc cancel generation",
      backgroundColor: theme.inputBg,
      focusedBackgroundColor: theme.inputBg,
      textColor: theme.text,
      placeholderColor: theme.muted,
    })
  ) as InputRenderable;

  const chatPane = mount(
    Box({
      width: "100%",
      height: "100%",
      flexDirection: "column",
      gap: 1,
      border: true,
      borderStyle: "rounded",
      borderColor: theme.border,
      backgroundColor: theme.panel,
      padding: 1,
      title: " Chat ",
      titleColor: theme.accent,
    })
  ) as BoxRenderable;
  chatPane.add(chatScroll);
  chatPane.add(chatInput);

  const panes: Record<PaneId, BoxRenderable> = {
    status: statusPane,
    backend: backendPane,
    model: modelPane,
    load: loadPane,
    chat: chatPane,
  };

  for (const pane of Object.values(panes)) {
    pane.visible = false;
    body.add(pane);
  }

  function focusPrimary(): void {
    if (activePane === "status") {
      statusActions.focus();
    } else if (activePane === "backend") {
      backendSelect.focus();
    } else if (activePane === "model") {
      if (modelBrowse === "hf-query") {
        modelSearch.focus();
      } else {
        modelSelect.focus();
      }
    } else if (activePane === "load") {
      loadFieldSelect.focus();
    } else {
      chatInput.focus();
    }
  }

  function cyclePane(delta: number): void {
    const i = PANE_ORDER.indexOf(activePane);
    const next = PANE_ORDER[(i + delta + PANE_ORDER.length) % PANE_ORDER.length];
    showPane(next);
  }

  /** Move focus between list and action sections (Backend / Load). */
  function shiftSectionFocus(dir: "left" | "right"): boolean {
    if (activePane === "backend") {
      if (dir === "right" && backendSelect.focused) {
        backendActions.focus();
        return true;
      }
      if (dir === "left" && backendActions.focused) {
        backendSelect.focus();
        return true;
      }
    } else if (activePane === "model") {
      if (modelBrowse === "hf-query") {
        if (dir === "right" && modelSearch.focused) {
          modelSelect.focus();
          return true;
        }
        if (dir === "left" && modelSelect.focused) {
          modelSearch.focus();
          return true;
        }
      }
    }
    return false;
  }

  function showPane(id: PaneId): void {
    if (loadEditing) {
      loadEditing = false;
    }
    activePane = id;
    for (const [key, pane] of Object.entries(panes) as Array<[PaneId, BoxRenderable]>) {
      pane.visible = key === id;
    }
    const idx = PANE_TABS.findIndex((t) => t.value === id);
    // Avoid setSelectedIndex → SELECTION_CHANGED → showPane recursion.
    if (idx >= 0 && tabs.getSelectedIndex() !== idx) {
      tabs.setSelectedIndex(idx);
    }
    refreshAll();
    focusPrimary();
    renderer.requestRender();
  }

  function setStatusMessage(msg: string): void {
    statusMessage = msg;
    refreshStatus();
    renderer.requestRender();
  }

  function renderChatLog(): void {
    if (!chatHistory.length) {
      chatLog.content = "Select a model, start the server, then type below.";
      return;
    }
    chatLog.content = chatHistory
      .map((m) => {
        const label = m.role === "user" ? "You" : m.role === "assistant" ? "Assistant" : "System";
        return `${label}:\n${m.content}\n`;
      })
      .join("\n");
  }

  function refreshStatus(): void {
    const st = services.processManager.getStatus();
    const state = services.store.getState();
    const color = statusColor(st.running, !!st.starting, !!st.configDirty);
    headerStatus.content = `${statusLabel(st.running, !!st.starting, !!st.configDirty)} · ${st.endpoint}`;
    headerStatus.fg = color as never;

    const lines = [
      `Endpoint:  ${st.endpoint}`,
      `PID:       ${st.pid ?? "—"}`,
      `Model:     ${shortPath(st.modelPath || state.selectedModelPath || "")}`,
      `Vision:    ${
        state.loadSettings.mmprojPath
          ? `${path.basename(state.loadSettings.mmprojPath)}${
              state.loadSettings.mmprojOffloadToGpu === false ? " · CPU" : " · GPU"
            }`
          : "off"
      }`,
      `Config:    ${services.config.path}`,
      `Dirty:     ${st.configDirty ? "yes — reload to apply" : "no"}`,
      `Owned:     ${st.ownedByThisExtension ? "yes" : "no / foreign"}`,
      "",
      bootMessage ? `Progress:  ${bootMessage}` : "",
      statusMessage ? `Note:      ${statusMessage}` : "",
      st.message ? `Server:    ${st.message}` : "",
    ].filter((l) => l !== undefined);

    statusInfo.content = lines.filter((l, i, a) => !(l === "" && a[i - 1] === "")).join("\n");
  }

  function refreshBackend(): void {
    const opts = services.installer.getUiBackendOptions();
    const info = services.installer.getInstalledInfo();
    backendSelect.options = opts.map((o) => {
      let desc = o.reason || "";
      if (o.id === "path") {
        desc = o.installed
          ? `Found on PATH${o.installedTag ? ` · ${o.installedTag}` : ""}`
          : o.reason || "llama-server not on PATH";
      } else if (o.installed) {
        desc = `Installed${o.installedTag ? ` ${o.installedTag}` : ""}${o.active ? " · active" : ""}`;
      } else if (o.available) {
        desc = "Not installed — Install will download";
      }
      return {
        name: `${o.label}${o.active ? " ●" : ""}`,
        description: desc,
        value: o.id,
      } satisfies SelectOption;
    });
    const activeIdx = opts.findIndex((o) => o.active);
    if (activeIdx >= 0) {
      backendSelect.setSelectedIndex(activeIdx);
    }

    const bits = [
      `Active: ${info.activeBackend}`,
      info.binaryVersion ? `Version: ${info.binaryVersion}` : "",
      info.binaryRunnable === false
        ? `Not runnable${info.nixOs ? " (NixOS / FHS)" : ""}${info.binaryRunError ? `: ${info.binaryRunError.slice(0, 80)}` : ""}`
        : "",
      info.pathBinary ? `PATH: ${info.pathBinary}` : "",
      statusMessage,
    ].filter(Boolean);
    backendHint.content = bits.join("\n");
  }

  function setModelBrowse(mode: ModelBrowseMode): void {
    modelBrowse = mode;
    modelSearch.visible = mode === "hf-query";
    if (mode === "local") {
      hfRepos = [];
      hfLicenses = new Map();
      hfPickedRepo = undefined;
      hfPickedLicense = undefined;
      hfFiles = [];
      hfAllFiles = [];
      modelSearch.value = "";
      refreshModels();
      modelSelect.focus();
    } else if (mode === "hf-query") {
      modelHint.content =
        "Search Hugging Face for GGUF repos. Enter runs search · Esc back to local library.";
      modelSelect.options = [];
      modelSearch.visible = true;
      modelSearch.focus();
    }
    renderer.requestRender();
  }

  function refreshModels(): void {
    if (modelBrowse !== "local") {
      return;
    }
    const entries = listLocalModelEntries(services.store.getConfig());
    const selected = services.store.getState().selectedModelPath;
    modelSelect.options = entries.map((e) => ({
      name: path.basename(e.path),
      description: `${e.source} · ${formatModelSize(e.sizeBytes)} · ${shortPath(e.path, 48)}`,
      value: e.path,
    }));
    if (!entries.length) {
      modelHint.content =
        "No GGUFs yet. Press / to search Hugging Face, or d for the starter model.";
    } else {
      modelHint.content = `Selected: ${shortPath(selected || "(none)")} · ${entries.length} local · Enter apply · / HF search · d starter`;
    }
    const idx = entries.findIndex((e) => e.path === selected);
    if (idx >= 0) {
      modelSelect.setSelectedIndex(idx);
    }
  }

  function showHfRepos(): void {
    modelBrowse = "hf-repos";
    modelSearch.visible = false;
    modelSelect.options = hfRepos.map((m) => {
      const license = hfLicenses.get(m.id) || licenseFromTags(m.tags);
      const row = formatLicenseQuickPick(m.id, m.downloads, license);
      return {
        name: row.label,
        description: row.description,
        value: m.id,
      };
    });
    modelHint.content = `${hfRepos.length} Hugging Face repo(s) · Enter pick · Esc back`;
    modelSelect.focus();
    renderer.requestRender();
  }

  function showHfConfirm(): void {
    modelBrowse = "hf-confirm";
    modelSearch.visible = false;
    const lic = hfPickedLicense;
    modelSelect.options = [
      {
        name: "Download anyway",
        description: lic?.summary || "License needs review",
        value: "confirm",
      },
      {
        name: "Cancel",
        description: "Back to search results",
        value: "cancel",
      },
    ];
    modelHint.content = `${hfPickedRepo?.id}\n${lic?.badge || "license"} · ${lic?.summary || ""}`;
    modelSelect.focus();
    renderer.requestRender();
  }

  function showHfFiles(): void {
    modelBrowse = "hf-files";
    modelSearch.visible = false;
    modelSelect.options = hfFiles.map((f) => ({
      name: f.path,
      description: formatBytes(f.size),
      value: f.path,
    }));
    const mmproj = preferredMmprojFile(hfAllFiles);
    const mtp = preferredMtpDraftFile(hfAllFiles);
    const extras = [mmproj?.path, mtp?.path].filter(Boolean).map((p) => path.basename(p as string));
    modelHint.content = extras.length
      ? `${hfPickedRepo?.id} · ${hfFiles.length} GGUF · also fetches ${extras.join(", ")} · Enter download · Esc back`
      : `${hfPickedRepo?.id} · ${hfFiles.length} GGUF file(s) · Enter download · Esc back`;
    modelSelect.focus();
    renderer.requestRender();
  }

  async function runHfSearch(query: string): Promise<void> {
    await withBusy("Searching Hugging Face…", async () => {
      hfRepos = await services.hf.searchGgufModels(query);
      if (!hfRepos.length) {
        setStatusMessage("No GGUF models found for that query.");
        setModelBrowse("hf-query");
        return;
      }
      bootMessage = "Resolving licenses…";
      refreshStatus();
      renderer.requestRender();
      hfLicenses = await services.hf.enrichLicenses(hfRepos);
      showHfRepos();
    });
  }

  async function pickHfRepo(modelId: string): Promise<void> {
    const model = hfRepos.find((m) => m.id === modelId);
    if (!model) {
      return;
    }
    hfPickedRepo = model;
    hfPickedLicense = hfLicenses.get(model.id) || licenseFromTags(model.tags);
    if (hfPickedLicense.needsConfirm) {
      showHfConfirm();
      return;
    }
    await loadHfFiles();
  }

  async function loadHfFiles(): Promise<void> {
    if (!hfPickedRepo) {
      return;
    }
    await withBusy("Listing GGUF files…", async () => {
      const all = await services.hf.listGgufFiles(hfPickedRepo!.id);
      hfFiles = languageGgufFiles(all);
      hfAllFiles = all;
      if (!hfFiles.length) {
        setStatusMessage("No language GGUF files in that repo.");
        showHfRepos();
        return;
      }
      showHfFiles();
    });
  }

  async function downloadHfFile(filePath: string): Promise<void> {
    if (!hfPickedRepo) {
      return;
    }
    const repoId = hfPickedRepo.id;
    await withBusy(`Downloading ${path.basename(filePath)}…`, async () => {
      const dest = await services.hf.downloadModelFile(repoId, filePath, {
        report: (v: { message?: string; increment?: number }) => {
          bootMessage = v.message || bootMessage;
          refreshStatus();
          renderer.requestRender();
        },
      });
      try {
        await services.hf.downloadPreferredMmproj(repoId, hfAllFiles, {
          report: (v: { message?: string; increment?: number }) => {
            bootMessage = v.message || bootMessage;
            refreshStatus();
            renderer.requestRender();
          },
        });
      } catch {
        // Language GGUF is enough to start; projector can be added later.
      }
      try {
        await services.hf.downloadPreferredMtpDraft(repoId, hfAllFiles, {
          report: (v: { message?: string; increment?: number }) => {
            bootMessage = v.message || bootMessage;
            refreshStatus();
            renderer.requestRender();
          },
        });
      } catch {
        // Optional sidecar MTP drafter.
      }
      await services.store.applySelectedModel(dest, {
        recommendDefaults: true,
        cpuOnly: services.installer.resolveActiveUiBackend() === "cpu",
        attachMmproj: true,
      });
      setStatusMessage(`Downloaded ${path.basename(dest)}. Start/reload to load it.`);
      setModelBrowse("local");
    });
  }

  async function downloadStarter(): Promise<void> {
    await withBusy(`Downloading starter ${STARTER_MODEL.label}…`, async () => {
      const dest = await services.hf.downloadModelFile(
        STARTER_MODEL.repoId,
        STARTER_MODEL.filePath,
        {
          report: (v: { message?: string; increment?: number }) => {
            bootMessage = v.message || bootMessage;
            refreshStatus();
            renderer.requestRender();
          },
        }
      );
      try {
        const files = await services.hf.listGgufFiles(STARTER_MODEL.repoId);
        await services.hf.downloadPreferredMmproj(STARTER_MODEL.repoId, files, {
          report: (v: { message?: string; increment?: number }) => {
            bootMessage = v.message || bootMessage;
            refreshStatus();
            renderer.requestRender();
          },
        });
        await services.hf.downloadPreferredMtpDraft(STARTER_MODEL.repoId, files, {
          report: (v: { message?: string; increment?: number }) => {
            bootMessage = v.message || bootMessage;
            refreshStatus();
            renderer.requestRender();
          },
        });
      } catch {
        // optional
      }
      await services.store.applySelectedModel(dest, {
        recommendDefaults: true,
        cpuOnly: services.installer.resolveActiveUiBackend() === "cpu",
        attachMmproj: true,
      });
      setStatusMessage(`Starter ready: ${path.basename(dest)}.`);
      setModelBrowse("local");
    });
  }

  function modelBrowseBack(): void {
    if (modelBrowse === "hf-files" || modelBrowse === "hf-confirm") {
      showHfRepos();
      return;
    }
    if (modelBrowse === "hf-repos" || modelBrowse === "hf-query") {
      setModelBrowse("local");
      return;
    }
  }

  function setVram2Visible(show: boolean): void {
    loadVram2Head.row.height = show ? 1 : 0;
    loadVram2Bar.row.height = show ? 1 : 0;
    if (!show) {
      loadVram2Head.title.content = "";
      loadVram2Head.sub.content = "";
      paintMemBar(loadVram2Bar, null, 0);
    }
  }

  function refreshMemoryCharts(): void {
    const state = services.store.getState();
    const caps = state.modelCapabilities;
    const barWidth = Math.max(
      24,
      Math.min(MEM_BAR_WIDTH, Math.max(24, (renderer.width || 80) - 8))
    );

    if (!caps) {
      loadVramHead.title.content = "VRAM · est. at full context";
      loadRamHead.title.content = "System RAM · est. at full context";
      loadVramHead.sub.content = "—";
      loadVramHead.sub.fg = theme.muted as never;
      loadRamHead.sub.content = "—";
      loadRamHead.sub.fg = theme.muted as never;
      paintMemBar(loadVramBar, null, barWidth);
      paintMemBar(loadRamBar, null, barWidth);
      setVram2Visible(false);
      loadMemSummary.content = "Select a model to estimate VRAM / RAM use.";
      loadMemSummary.fg = theme.muted as never;
      loadMemWarn.content = "";
      return;
    }

    const cpuOnly = services.installer.resolveActiveUiBackend() === "cpu";
    const gpus = cpuOnly ? [] : detectGpus(false, services.processManager.resolveBinary());
    const gpu = gpus[0];
    const est = estimateMemory(caps, state.loadSettings, gpu, {
      cpuOnly,
      gpus: cpuOnly ? undefined : gpus,
    });
    if (!est) {
      loadMemSummary.content = "Memory estimate unavailable.";
      loadMemWarn.content = "";
      paintMemBar(loadVramBar, null, barWidth);
      paintMemBar(loadRamBar, null, barWidth);
      setVram2Visible(false);
      return;
    }

    if (cpuOnly) {
      loadVramHead.title.content = "VRAM · CPU backend (unused)";
      loadVramHead.sub.content = "—";
      loadVramHead.sub.fg = theme.muted as never;
      paintMemBar(loadVramBar, null, barWidth);
      setVram2Visible(false);
    } else {
      loadVramHead.title.content = est.charts.vram.title;
      const vSub = chartSubtitle(est.charts.vram);
      loadVramHead.sub.content = vSub.text;
      loadVramHead.sub.fg = memToneFg(vSub.tone) as never;
      paintMemBar(loadVramBar, est.charts.vram, barWidth);
      if (est.charts.vram2) {
        setVram2Visible(true);
        loadVram2Head.title.content = est.charts.vram2.title;
        const v2Sub = chartSubtitle(est.charts.vram2);
        loadVram2Head.sub.content = v2Sub.text;
        loadVram2Head.sub.fg = memToneFg(v2Sub.tone) as never;
        paintMemBar(loadVram2Bar, est.charts.vram2, barWidth);
      } else {
        setVram2Visible(false);
      }
    }

    loadRamHead.title.content = est.charts.ram.title;
    const rSub = chartSubtitle(est.charts.ram);
    loadRamHead.sub.content = rSub.text;
    loadRamHead.sub.fg = memToneFg(rSub.tone) as never;
    paintMemBar(loadRamBar, est.charts.ram, barWidth);

    loadMemSummary.content = est.summary;
    loadMemSummary.fg = (est.willSpill ? theme.bad : theme.text) as never;

    const warnLine = est.warnings[0] || "";
    loadMemWarn.content = warnLine;
    loadMemWarn.fg = (est.willSpill ? theme.bad : theme.warn) as never;
  }

  function refreshLoad(): void {
    const state = services.store.getState();
    const load = state.loadSettings;
    const req = state.requestSettings;
    const selectedId = loadFieldSelect.getSelectedOption()?.value as string | undefined;

    syncMainGpuOptions();
    maybeMigrateLegacySplit();
    refreshMemoryCharts();
    loadHint.content = [
      `ctx ${load.contextLength} · ngl ${load.gpuOffload} · KV ${load.cacheTypeK}/${load.cacheTypeV} · slots ${load.maxConcurrentPredictions}`,
      `sampling T=${req.temperature} top_p=${req.topP} top_k=${req.topK} max_tokens=${req.maxTokens}`,
    ].join("\n");

    loadFieldSelect.options = visibleLoadFields().map((f) => formatLoadOption(f));
    if (selectedId) {
      const idx = loadFieldSelect.options.findIndex((o) => o.value === selectedId);
      if (idx >= 0) {
        loadFieldSelect.setSelectedIndex(idx);
      }
    }
    refreshLoadDetail();
  }

  function refreshAll(): void {
    refreshStatus();
    if (activePane === "backend") {
      refreshBackend();
    }
    if (activePane === "model") {
      refreshModels();
    }
    if (activePane === "load") {
      refreshLoad();
    }
    if (activePane === "chat") {
      renderChatLog();
    }
  }

  async function withBusy(label: string, fn: () => Promise<void>): Promise<void> {
    if (busy) {
      setStatusMessage("Busy — wait for the current action.");
      return;
    }
    busy = true;
    bootMessage = label;
    statusMessage = label;
    refreshStatus();
    if (activePane === "model") {
      modelHint.content = label;
    }
    renderer.requestRender();
    try {
      // Paint the busy state before sync work (GGUF reads, etc.) blocks the loop.
      await renderer.idle();
      await new Promise<void>((resolve) => setImmediate(resolve));
      await fn();
    } catch (err) {
      setStatusMessage(err instanceof Error ? err.message : String(err));
    } finally {
      busy = false;
      bootMessage = "";
      refreshAll();
      renderer.requestRender();
    }
  }

  async function startServer(): Promise<void> {
    await withBusy("Starting…", async () => {
      const token = services.processManager.claimLaunch("start");
      try {
        const status = await services.processManager.start(
          undefined,
          (msg) => {
            bootMessage = msg;
            refreshStatus();
            renderer.requestRender();
          },
          token ?? undefined
        );
        setStatusMessage(status.message || "Server started.");
      } finally {
        if (token) {
          services.processManager.releaseLaunch(token);
        }
      }
    });
  }

  async function stopServer(): Promise<void> {
    await withBusy("Stopping…", async () => {
      await services.processManager.stop(true);
      setStatusMessage("Server stopped.");
    });
  }

  async function reloadServer(): Promise<void> {
    await withBusy("Reloading…", async () => {
      const token = services.processManager.claimLaunch("reload");
      try {
        const status = await services.processManager.reload((msg) => {
          bootMessage = msg;
          refreshStatus();
          renderer.requestRender();
        }, token ?? undefined);
        setStatusMessage(status.message || "Server reloaded.");
      } finally {
        if (token) {
          services.processManager.releaseLaunch(token);
        }
      }
    });
  }

  async function useBackend(id: UiBackend): Promise<void> {
    await withBusy(`Switching to ${id}…`, async () => {
      await services.installer.setBackend(id);
      if (id === "path" && !services.installer.hasBackendInstalled("path")) {
        setStatusMessage("llama-server not found on PATH.");
        return;
      }
      setStatusMessage(`Backend set to ${id}. Reload/start to use it.`);
    });
  }

  async function installBackend(id: UiBackend): Promise<void> {
    if (id === "path") {
      await useBackend("path");
      return;
    }
    await withBusy(`Installing ${id}…`, async () => {
      await services.installer.setBackend(id);
      await services.installer.installOrUpgrade(
        {
          report: (v: { message?: string; increment?: number }) => {
            bootMessage = v.message || bootMessage;
            refreshStatus();
            renderer.requestRender();
          },
        },
        id
      );
      setStatusMessage(`Installed ${id}.`);
    });
  }

  async function selectModel(modelPath: string, recommend: boolean): Promise<void> {
    const name = path.basename(modelPath);
    await withBusy(`Reading ${name}…`, async () => {
      await services.store.applySelectedModel(modelPath, {
        recommendDefaults: recommend,
        cpuOnly: services.installer.resolveActiveUiBackend() === "cpu",
      });
      setStatusMessage(`Selected ${name}. Start/reload to load it.`);
    });
  }

  function fittingContext(maxCtx: number, load: LlamaLoadSettings): number {
    const state = services.store.getState();
    const caps = state.modelCapabilities;
    if (!caps) {
      return Math.min(65536, maxCtx);
    }
    const cpuOnly = services.installer.resolveActiveUiBackend() === "cpu";
    const gpus = cpuOnly ? [] : detectGpus(false, services.processManager.resolveBinary());
    if (cpuOnly || !gpus.length) {
      return Math.min(65536, maxCtx);
    }
    const gpu = gpus[0];
    let best = 0;
    for (const step of FIT_CONTEXT_STEPS) {
      const ctx = Math.min(step, maxCtx);
      if (ctx < 8192) {
        continue;
      }
      const est = estimateMemory(caps, { ...load, contextLength: ctx }, gpu, {
        cpuOnly,
        gpus,
      });
      if (est && !est.willSpill) {
        best = ctx;
        break;
      }
    }
    return best || Math.min(8192, maxCtx);
  }

  async function applyPreset(id: string): Promise<void> {
    const preset = LOAD_PRESETS[id];
    if (!preset) {
      return;
    }
    await withBusy(`Preset: ${preset.label}`, async () => {
      const state = services.store.getState();
      const maxCtx = state.modelCapabilities?.maxContextLength || 131072;
      const patch: Partial<LlamaLoadSettings> = {
        cacheTypeK: preset.cacheTypeK,
        cacheTypeV: preset.cacheTypeV,
        maxConcurrentPredictions: 1,
      };
      const base = { ...state.loadSettings, ...patch };
      patch.contextLength =
        preset.contextLength === "fit"
          ? fittingContext(maxCtx, base)
          : Math.min(preset.contextLength, maxCtx);
      await services.store.updateLoadSettings(patch);
      setStatusMessage(`Applied ${preset.label}. Reload to use.`);
    });
  }

  async function sendChat(text: string): Promise<void> {
    const prompt = text.trim();
    if (!prompt || chatBusy) {
      return;
    }
    const ready = await services.processManager.isHttpReady();
    if (!ready) {
      setStatusMessage("Server is not ready — start it on the Status pane.");
      showPane("status");
      return;
    }

    chatBusy = true;
    chatAbort = new AbortController();
    chatHistory.push({ role: "user", content: prompt });
    chatHistory.push({ role: "assistant", content: "" });
    renderChatLog();
    renderer.requestRender();

    const req = services.store.getState().requestSettings;
    try {
      for await (const ev of streamChatCompletion({
        endpoint: services.store.getEndpoint(),
        messages: chatHistory.slice(0, -1),
        temperature: req.temperature,
        topP: req.topP,
        topK: req.topK,
        maxTokens: req.maxTokens,
        signal: chatAbort.signal,
      })) {
        if (ev.kind === "text") {
          const last = chatHistory[chatHistory.length - 1];
          if (last?.role === "assistant") {
            last.content += ev.text;
            renderChatLog();
            renderer.requestRender();
          }
        } else if (ev.kind === "stats") {
          const bits = [
            ev.genTokPerSec ? `${ev.genTokPerSec.toFixed(1)} tok/s` : "",
            ev.promptTokens != null ? `prompt ${ev.promptTokens}` : "",
            ev.cachedPromptTokens != null && ev.processedPromptTokens != null
              ? `cache ${ev.cachedPromptTokens}/${ev.cachedPromptTokens + ev.processedPromptTokens}`
              : "",
          ].filter(Boolean);
          if (bits.length) {
            setStatusMessage(bits.join(" · "));
          }
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg !== "Cancelled") {
        const last = chatHistory[chatHistory.length - 1];
        if (last?.role === "assistant" && !last.content) {
          last.content = `(error: ${msg})`;
        } else {
          chatHistory.push({ role: "assistant", content: `(error: ${msg})` });
        }
        setStatusMessage(msg);
      }
    } finally {
      chatBusy = false;
      chatAbort = undefined;
      renderChatLog();
      chatInput.focus();
      renderer.requestRender();
    }
  }

  // ─── events ────────────────────────────────────────────────────────────
  tabs.on(TabSelectRenderableEvents.SELECTION_CHANGED, () => {
    const opt = tabs.getSelectedOption();
    if (opt?.value && opt.value !== activePane) {
      showPane(opt.value as PaneId);
    }
  });

  statusActions.on(SelectRenderableEvents.ITEM_SELECTED, () => {
    const v = statusActions.getSelectedOption()?.value;
    if (v === "start") {
      void startServer();
    } else if (v === "stop") {
      void stopServer();
    } else if (v === "reload") {
      void reloadServer();
    } else if (v === "refresh") {
      refreshAll();
      renderer.requestRender();
    }
  });

  backendSelect.on(SelectRenderableEvents.ITEM_SELECTED, () => {
    const id = backendSelect.getSelectedOption()?.value as UiBackend | undefined;
    if (id) {
      void useBackend(id);
    }
  });

  backendActions.on(SelectRenderableEvents.ITEM_SELECTED, () => {
    const action = backendActions.getSelectedOption()?.value;
    const id = backendSelect.getSelectedOption()?.value as UiBackend | undefined;
    if (!id) {
      return;
    }
    if (action === "use") {
      void useBackend(id);
    } else if (action === "install") {
      void installBackend(id);
    } else if (action === "refresh") {
      refreshBackend();
      renderer.requestRender();
    }
  });

  modelSelect.on(SelectRenderableEvents.ITEM_SELECTED, () => {
    const value = modelSelect.getSelectedOption()?.value as string | undefined;
    if (!value) {
      return;
    }
    if (modelBrowse === "local") {
      void selectModel(value, true);
      return;
    }
    if (modelBrowse === "hf-repos") {
      void pickHfRepo(value);
      return;
    }
    if (modelBrowse === "hf-confirm") {
      if (value === "confirm") {
        void loadHfFiles();
      } else {
        showHfRepos();
      }
      return;
    }
    if (modelBrowse === "hf-files") {
      void downloadHfFile(value);
    }
  });

  modelSearch.on(InputRenderableEvents.ENTER, () => {
    const q = modelSearch.value.trim();
    void runHfSearch(q || "gguf");
  });

  loadFieldSelect.on(SelectRenderableEvents.SELECTION_CHANGED, () => {
    // Navigating the list always leaves edit mode.
    if (loadEditing) {
      loadEditing = false;
    }
    refreshLoadDetail();
    renderer.requestRender();
  });

  loadFieldSelect.on(SelectRenderableEvents.ITEM_SELECTED, () => {
    activateLoadField();
  });

  chatInput.on(InputRenderableEvents.ENTER, () => {
    const text = chatInput.value;
    chatDraft = "";
    chatInput.value = "";
    void sendChat(text);
  });

  chatInput.on(InputRenderableEvents.INPUT, () => {
    chatDraft = chatInput.value;
  });

  services.store.onDidChange(() => {
    refreshAll();
    renderer.requestRender();
  });

  renderer.keyInput.on("keypress", (key) => {
    if (key.name === "f12") {
      key.preventDefault();
      void reloadServer();
      return;
    }

    if (key.name === "escape") {
      if (activePane === "load" && loadEditing) {
        key.preventDefault();
        setLoadEditing(false);
        return;
      }
      if (chatBusy) {
        chatAbort?.abort();
        return;
      }
      if (activePane === "model" && modelBrowse !== "local") {
        modelBrowseBack();
        return;
      }
    }

    // Load edit mode: ←→/↑↓ adjust, Enter/Esc done (list is blurred while editing).
    if (activePane === "load" && loadEditing && !key.ctrl) {
      if (key.name === "left" || key.name === "right") {
        key.preventDefault();
        adjustLoadField(key.name === "right" ? 1 : -1);
        return;
      }
      if (key.name === "up" || key.name === "down") {
        key.preventDefault();
        adjustLoadField(key.name === "down" ? 1 : -1);
        return;
      }
      if (key.name === "return" || key.name === "enter") {
        key.preventDefault();
        setLoadEditing(false);
        return;
      }
    }

    // Panel navigation works everywhere — including while typing in Chat.
    const fMap: Record<string, PaneId> = {
      f1: "status",
      f2: "backend",
      f3: "model",
      f4: "load",
      f5: "chat",
    };
    if (key.name && fMap[key.name]) {
      showPane(fMap[key.name]);
      return;
    }
    if (key.name === "tab") {
      cyclePane(key.shift ? -1 : 1);
      return;
    }

    // ←→ moves focus between list and action sections (not while editing text / load values).
    if (
      (key.name === "left" || key.name === "right") &&
      !(activePane === "chat" && chatInput.focused) &&
      !(activePane === "model" && modelSearch.focused) &&
      !(activePane === "load")
    ) {
      if (shiftSectionFocus(key.name === "right" ? "right" : "left")) {
        return;
      }
    }

    // While typing in chat or HF search, don't steal digits / letters.
    if (activePane === "chat" && chatInput.focused) {
      return;
    }
    if (activePane === "model" && modelSearch.focused) {
      return;
    }

    // Model pane shortcuts
    if (activePane === "model" && modelBrowse === "local" && !key.ctrl) {
      if (key.name === "/" || key.sequence === "/") {
        setModelBrowse("hf-query");
        return;
      }
      if (key.name === "d") {
        void downloadStarter();
        return;
      }
      if (key.name === "g") {
        refreshModels();
        renderer.requestRender();
        return;
      }
    }

    const digitMap: Record<string, PaneId> = {
      "1": "status",
      "2": "backend",
      "3": "model",
      "4": "load",
      "5": "chat",
    };
    if (key.name && digitMap[key.name]) {
      showPane(digitMap[key.name]);
      return;
    }
    if (key.name === "s" && !key.ctrl) {
      void startServer();
      return;
    }
    if (key.name === "x" && !key.ctrl) {
      void stopServer();
      return;
    }
    if (key.name === "r" && !key.ctrl) {
      void reloadServer();
      return;
    }
    if (key.name === "q" && !key.ctrl) {
      renderer.destroy();
      process.exit(0);
    }
  });

  // Keep header fresh while starting
  const timer = setInterval(() => {
    if (busy || services.processManager.getStatus().starting) {
      refreshStatus();
      renderer.requestRender();
    }
  }, 500);

  renderer.on("destroy", () => {
    clearInterval(timer);
    chatAbort?.abort();
  });

  showPane("status");
  void chatDraft;
}
