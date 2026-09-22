/** Placeholder for image / binary tool-result parts. Counting or sending the raw bytes blows the context. */
export const OMITTED_BINARY_TOOL_RESULT = "[image omitted]";

/**
 * Join tool-result parts the way a chat template expects: text stays text,
 * binary and numeric byte-maps become {@link OMITTED_BINARY_TOOL_RESULT}.
 */
export function formatToolResultContent(content: readonly unknown[]): string {
  return content
    .map((item) => formatToolResultItem(item))
    .filter((item) => item && item !== "null" && item !== "undefined")
    .join("\n");
}

function formatToolResultItem(item: unknown): string {
  if (typeof item === "string") {
    return item;
  }
  if (!item || typeof item !== "object") {
    return "";
  }
  const rec = item as { value?: unknown; text?: unknown; data?: unknown; mimeType?: unknown };
  if (typeof rec.value === "string") {
    return rec.value;
  }
  if (typeof rec.text === "string") {
    return rec.text;
  }
  if (isBinaryToolPayload(rec)) {
    return OMITTED_BINARY_TOOL_RESULT;
  }
  try {
    const json = JSON.stringify(item);
    if (!json || json === "null" || json === "undefined") {
      return "";
    }
    if (jsonLooksLikeByteMap(json)) {
      return OMITTED_BINARY_TOOL_RESULT;
    }
    return json;
  } catch {
    return "";
  }
}

function isBinaryToolPayload(rec: { data?: unknown; value?: unknown; mimeType?: unknown }): boolean {
  if (isByteMap(rec.data) || isByteMap(rec.value)) {
    return true;
  }
  const mime = typeof rec.mimeType === "string" ? rec.mimeType : "";
  if (/^(image|audio|video)\//i.test(mime) || /^application\/octet-stream/i.test(mime)) {
    return true;
  }
  return false;
}

function isByteMap(value: unknown): boolean {
  if (!value || typeof value !== "object") {
    return false;
  }
  if (value instanceof Uint8Array) {
    return true;
  }
  if (typeof Buffer !== "undefined" && Buffer.isBuffer(value)) {
    return true;
  }
  if (Array.isArray(value)) {
    return value.length >= 8 && value.every((n) => typeof n === "number" && n >= 0 && n <= 255);
  }
  const entries = Object.entries(value as Record<string, unknown>);
  if (entries.length < 8) {
    return false;
  }
  return entries.every(
    ([key, entry]) => /^\d+$/.test(key) && typeof entry === "number" && entry >= 0 && entry <= 255
  );
}

function jsonLooksLikeByteMap(json: string): boolean {
  const compact = json.replace(/\s/g, "");
  return compact.length > 64 && /^\{(?:"\d+":\d+,?){8,}/.test(compact);
}
