/**
 * Messages between the sidebar webview (./sidebar.ts) and SettingsViewProvider.
 * Both sides import this file, so adding, renaming or removing a message is a
 * compile error on whichever side was not updated. Type-only: no runtime code.
 *
 * Load / request settings travel as loose records — the webview reads them
 * straight from form fields and the host validates through SettingsStore.
 */

type SettingsPatch = Record<string, unknown>;

/** Webview → extension host. */
export type ToHostMessage =
  // Lifecycle
  | { type: "ready" }
  | { type: "start"; payload?: SettingsPatch }
  | { type: "reload"; payload: SettingsPatch }
  | { type: "stop" }
  | { type: "openLog" }
  | { type: "copyCommandLine" }
  | { type: "setLaunchMode"; payload: "background" | "externalTerminal" }
  // Load / request settings
  | { type: "saveLoad"; payload: SettingsPatch; silent?: boolean }
  | { type: "saveRequest"; payload: SettingsPatch }
  | { type: "resetAdvancedLoad" }
  | { type: "resetRequestDefaults" }
  | { type: "estimate"; seq: number; payload: SettingsPatch; cpuOnly?: boolean }
  | { type: "applyLoadPatch"; payload: SettingsPatch; fitContext?: boolean; cpuOnly?: boolean }
  | { type: "discardChanges" }
  // Model, draft, projector, template pickers
  | { type: "downloadStarter" }
  | { type: "openModelFile" }
  | { type: "pickDraftModel" }
  | { type: "clearDraftModel" }
  | { type: "pickMmproj" }
  | { type: "clearMmproj" }
  | { type: "pickChatTemplate" }
  // llama.cpp install
  | { type: "installLlamaCpp"; payload?: string }
  | { type: "reinstallLlamaCpp" }
  | { type: "installLlamaCppByTag" }
  | { type: "installLlamaCppFromArchive" }
  | { type: "cancelLlamaInstall" }
  | { type: "retryLlamaInstall" }
  | { type: "checkUpdates" }
  | { type: "switchBackend"; payload: string }
  // Copilot request helpers
  | { type: "viewLastCall" }
  | { type: "viewLastResponse" }
  | { type: "setPromptReplacementsEnabled"; payload: { enabled: boolean } }
  | { type: "setWikipediaLookupEnabled"; payload: { enabled: boolean } }
  | { type: "setDuplicateToolCallGuardEnabled"; payload: { enabled: boolean } }
  // Remote server
  | { type: "refreshRemote" }
  | { type: "testRemote"; url: string; apiKey?: string }
  | { type: "connectRemote"; url: string; label?: string; apiKey?: string }
  | { type: "switchToLocal" }
  | { type: "setApiKey" }
  | { type: "forgetApiKey" }
  | { type: "copyDiagnostics" }
  // Model library
  | { type: "useLibraryModel"; path: string }
  | { type: "useLibraryDraft"; path: string }
  | { type: "copyPath"; path: string }
  | { type: "revealInOs"; path: string }
  | { type: "hideLibraryModel"; path: string }
  | { type: "unhideLibraryModel"; path: string }
  | { type: "removeLibraryQuant"; path: string; companionPaths: string[] }
  | { type: "cleanPartialDownloads" }
  // Hugging Face search and downloads
  | { type: "searchLibraryHf"; query: string }
  | { type: "listLibraryHfFiles"; modelId: string }
  | { type: "downloadLibraryHf"; modelId: string; filePath: string }
  | { type: "pauseDownload"; id: string }
  | { type: "resumeDownload"; id: string }
  | { type: "cancelDownload"; id: string }
  | { type: "setHfToken" }
  | { type: "openExternal"; url: string };

export type ToHostType = ToHostMessage["type"];

/**
 * Large view-model payloads (full state, status patch, memory view) are built
 * by the host from core types and rendered by untyped sidebar code, so their
 * inner shape is not modelled here; the message envelope is.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ViewModel = any;

/** Extension host → webview. */
export type ToWebviewMessage =
  | { type: "state"; payload: ViewModel }
  | { type: "statusPatch"; payload: ViewModel }
  | { type: "bootProgress"; payload: { starting: boolean; message: string; running: boolean; configDirty: boolean } }
  | {
      type: "perfPatch";
      payload: { perf: ViewModel; perfLines: string[]; hasLastContext: boolean; hasLastResponse: boolean };
    }
  | { type: "memoryEstimate"; seq: number; view: ViewModel }
  | {
      type: "llamaInstall";
      payload: {
        status: "idle" | "running" | "done" | "error" | "cancelled";
        phase?: string;
        message?: string;
        error?: string;
        percent?: number;
        received?: number;
        total?: number;
        bytesPerSec?: number;
        etaSeconds?: number;
        fileIndex?: number;
        fileCount?: number;
        assetName?: string;
        releaseTag?: string;
      };
    }
  | { type: "updateCheck"; payload: ViewModel }
  | { type: "draftModelSelected"; path: string }
  | { type: "mmprojSelected"; path: string }
  | { type: "openConnect" }
  | { type: "remoteProbe"; payload?: ViewModel; error?: string }
  | {
      type: "focusLibrary";
      filter?: boolean;
      downloads?: boolean;
      query?: string;
      search?: boolean;
    }
  | {
      type: "libraryFits";
      fits: Record<string, string>;
      ratios: Record<string, number>;
      drafts: Record<string, boolean>;
      budgetLabel: string;
    }
  | {
      type: "libraryDownloads";
      jobs: Array<{
        id: string;
        label: string;
        received: number;
        total: number;
        bytesPerSec?: number;
        etaSeconds?: number;
        state: string;
        error: string;
        pageUrl: string;
      }>;
    }
  | {
      type: "libraryHfResults";
      query: string;
      repos: Array<{
        id: string;
        downloads?: number;
        badge: string;
        needsConfirm: boolean;
        summary: string;
        licenseUrl: string;
      }>;
    }
  | { type: "libraryHfError"; query: string; message: string }
  | {
      type: "libraryHfFiles";
      modelId: string;
      companionOnly: boolean;
      companionHint: string;
      files: Array<{ path: string; label: string; detail: string }>;
    };

export type ToWebviewType = ToWebviewMessage["type"];
