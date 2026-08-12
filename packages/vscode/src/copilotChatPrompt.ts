import * as path from "path";
import * as vscode from "vscode";
import { SettingsStore } from "@llama-aio/core";

/**
 * After llama-server is ready, remind the user to pick our model in Copilot Chat
 * (VS Code has no API to set the Chat model picker automatically).
 */
export async function promptUseInCopilotChat(
  store: SettingsStore,
  statusMessage?: string
): Promise<void> {
  const modelPath = store.getState().selectedModelPath;
  const shortName = modelPath ? path.basename(modelPath) : "local model";
  const label = `Llama AIO: ${shortName}`;
  const base = (statusMessage || "").trim() || "Llama AIO server is ready.";

  const choice = await vscode.window.showInformationMessage(
    `${base}\nSelect “${label}” in the Copilot Chat model picker to use it.`,
    "Open Chat",
    "Open model picker"
  );

  if (!choice) {
    return;
  }

  try {
    await vscode.commands.executeCommand("workbench.action.chat.open");
  } catch {
    try {
      await vscode.commands.executeCommand("workbench.action.chat.focus");
    } catch {
      // Chat view may be unavailable (no Copilot / older VS Code).
    }
  }

  if (choice === "Open model picker") {
    try {
      await vscode.commands.executeCommand("workbench.action.chat.openModelPicker");
    } catch {
      // Picker command varies by VS Code version; Chat is already open.
    }
  }
}
