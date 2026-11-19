import { ConnectionError, joinServerUrl, serverJson } from "@llama-aio/core";

export function maySupportReasoningControl(buildInfo: string | undefined): boolean {
  const upstreamBuild = /^b(\d+)(?:-|$)/.exec(buildInfo ?? "");
  // Upstream added the control endpoint in b9468. Unknown builds can try it
  // at runtime, since custom builds do not always report an upstream tag.
  return !upstreamBuild || Number(upstreamBuild[1]) >= 9468;
}

export function isReasoningControlUnavailable(error: unknown): boolean {
  return error instanceof ConnectionError && (error.status === 404 || error.status === 405);
}

export async function requestReasoningEnd(
  endpoint: string, model: string, id: string, signal?: AbortSignal
): Promise<void> {
  const result = await serverJson<{ success?: boolean; message?: string }>(
    joinServerUrl(endpoint, "/v1/chat/completions/control"), {
      method: "POST", body: { id, model, action: "reasoning_end" }, timeoutMs: 5000, signal,
    });
  if (result.success !== true) throw new Error(result.message || "The server did not end thinking.");
}
