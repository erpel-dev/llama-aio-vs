/**
 * The webview's handle to VS Code, shared by every sidebar module. Messages
 * are typed by ./protocol, so a renamed or misspelled message fails the build.
 */
import type { ToHostMessage } from "./protocol";

declare function acquireVsCodeApi(): {
  postMessage(msg: ToHostMessage): void;
  getState(): any;
  setState(state: any): void;
};

export const vscode = acquireVsCodeApi();

// Untyped on purpose: the sidebar predates the TypeScript move and reads
// .value, .checked, .dataset … from many element kinds.
export const $ = (id: string): any => document.getElementById(id);
