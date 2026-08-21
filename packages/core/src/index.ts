/**
 * Public surface of the shared core. Both the VS Code extension and the TUI
 * build on exactly these pieces — nothing here may import a frontend.
 */

export * from "./chatClient";
export * from "./config";
export * from "./contextBreakdown";
export * from "./events";
export * from "./externalTerminal";
export * from "./ggufMetadata";
export * from "./gpuInfo";
export * from "./hfLicense";
export * from "./huggingFace";
export * from "./installSwap";
export * from "./llamaInstaller";
export * from "./llamaTimings";
export * from "./memoryEstimate";
export * from "./modelLibrary";
export * from "./modelModes";
export * from "./nixCompat";
export * from "./paths";
export * from "./perfStats";
export * from "./processIdentity";
export * from "./processManager";
export * from "./promptReplacer";
export * from "./recommendSettings";
export * from "./serverArgs";
export * from "./settings";
export * from "./sseStream";
export * from "./types";
