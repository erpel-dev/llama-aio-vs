/**
 * Hugging Face downloads for the sidebar library and the starter model.
 */
import * as vscode from "vscode";
import {
  attachDownloadedCompanion,
  downloadManager,
  downloadPickerFiles,
  huggingfaceModelPage,
  huggingfaceUrl,
  listingBaseName,
  preferredMmprojFile,
  preferredMtpDraftFile,
  remoteShardPaths,
  STARTER_MODEL,
  HuggingFaceClient,
  type HfFileHit,
  type SettingsStore,
} from "@llama-aio/core";

export { HuggingFaceClient, STARTER_MODEL };

function hfToken(store: SettingsStore): string | undefined {
  const t = (store.getConfig().get<string>("hfToken") || "").trim();
  return t || undefined;
}

function enqueueHfFile(
  hf: HuggingFaceClient,
  store: SettingsStore,
  modelId: string,
  filePath: string,
  files?: HfFileHit[]
) {
  const meta = files?.find((f) => f.path === filePath);
  return downloadManager.enqueue({
    label: listingBaseName(filePath),
    dest: hf.localDestFor(modelId, filePath),
    url: meta?.url || huggingfaceUrl(`/${modelId}/resolve/main/${filePath}`),
    token: hfToken(store),
    expectedSize: meta?.size && meta.size > 0 ? meta.size : undefined,
    expectedSha256: meta?.sha256,
    gatedPageUrl: huggingfaceModelPage(modelId),
    modelId,
  });
}

async function enqueueModelAndCompanions(
  hf: HuggingFaceClient,
  store: SettingsStore,
  modelId: string,
  languagePath: string,
  files?: HfFileHit[]
): Promise<string> {
  const parts = remoteShardPaths(languagePath);
  const jobs = parts.map((part) => enqueueHfFile(hf, store, modelId, part, files));
  const listing = files ?? (await hf.listGgufFiles(modelId).catch(() => undefined));
  const extras: ReturnType<typeof enqueueHfFile>[] = [];
  const mm = listing ? preferredMmprojFile(listing) : undefined;
  if (mm) {
    extras.push(enqueueHfFile(hf, store, modelId, mm.path, listing));
  }
  const mtp = listing ? preferredMtpDraftFile(listing, languagePath) : undefined;
  if (mtp) {
    extras.push(enqueueHfFile(hf, store, modelId, mtp.path, listing));
  }
  // Companions are optional: a gated or missing mmproj / MTP file must not
  // throw away a finished language model.
  const companions = Promise.allSettled(extras.map((j) => j.done));
  let dests: string[];
  try {
    dests = await Promise.all(jobs.map((j) => j.done));
  } catch (e) {
    // One shard failed: the model cannot load, so stop spending bandwidth on
    // the rest. Pause keeps their partial files for a later resume.
    for (const j of [...jobs, ...extras]) {
      downloadManager.pause(j.job.id);
    }
    throw e;
  }
  const failed = (await companions).flatMap((r, i) =>
    r.status === "rejected" ? [`${extras[i].job.label}: ${r.reason instanceof Error ? r.reason.message : String(r.reason)}`] : []
  );
  if (failed.length) {
    void vscode.window.showWarningMessage(
      `Model downloaded, but ${failed.length === 1 ? "a companion file" : `${failed.length} companion files`} failed: ${failed.join("; ")}`
    );
  }
  return dests[0] || hf.localDestFor(modelId, parts[0]!);
}

/**
 * Download a GGUF chosen in the library. Progress stays on the library rows.
 * Returns the language-model path, or undefined when the file was a companion.
 */
export async function downloadGgufSelection(
  hf: HuggingFaceClient,
  store: SettingsStore,
  modelId: string,
  filePath: string
): Promise<string | undefined> {
  const files = await hf.listGgufFiles(modelId);
  const picker = downloadPickerFiles(files);
  const picked = picker.files.find((f) => f.path === filePath);
  if (!picked) {
    throw new Error("That GGUF is no longer listed in the repo.");
  }
  if (picker.companionOnly) {
    const dest = await enqueueHfFile(hf, store, modelId, picked.path, files).done;
    await attachDownloadedCompanion(store, dest, picked.path, files);
    return undefined;
  }
  return enqueueModelAndCompanions(hf, store, modelId, picked.path, files);
}

/**
 * One-click download of the curated starter GGUF via the HF resolve URL
 * (no browse/search). Reuses an existing local copy if present.
 */
export async function downloadStarterModel(
  hf: HuggingFaceClient,
  store: SettingsStore
): Promise<string | undefined> {
  const dest = await enqueueModelAndCompanions(
    hf,
    store,
    STARTER_MODEL.repoId,
    STARTER_MODEL.filePath
  );

  await store.applySelectedModel(dest, { attachMmproj: true });
  return dest;
}
