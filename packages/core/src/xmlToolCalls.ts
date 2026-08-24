export interface XmlToolCall {
  name: string;
  input: object;
  raw: string;
}

function decodeParamValue(raw: string): unknown {
  const trimmed = raw.trim();
  if (!trimmed) {
    return "";
  }
  try {
    return JSON.parse(trimmed);
  } catch {
    return trimmed;
  }
}

function parseParamTags(body: string): Record<string, unknown> {
  const input: Record<string, unknown> = {};
  const paramRe = /<parameter=([^>\n]+)>\s*([\s\S]*?)\s*<\/parameter>/gi;
  let p: RegExpExecArray | null;
  while ((p = paramRe.exec(body)) !== null) {
    input[p[1]!.trim()] = decodeParamValue(p[2]!);
  }
  const argRe = /<arg_key>\s*([\s\S]*?)\s*<\/arg_key>\s*<arg_value>([\s\S]*?)<\/arg_value>/gi;
  while ((p = argRe.exec(body)) !== null) {
    input[p[1]!.trim()] = decodeParamValue(p[2]!);
  }
  return input;
}

function toolNameFromOpenTag(attrs: string): string {
  const m = /\bname\s*=\s*["']([^"']+)["']/i.exec(attrs);
  return m?.[1]?.trim() || "";
}

function toolNameFromBody(body: string): string {
  const fn = /<function=([^>\n]+)>/i.exec(body);
  if (fn) {
    return fn[1]!.trim();
  }
  const invoke = /<invoke\s+name\s*=\s*["']([^"']+)["']/i.exec(body);
  if (invoke) {
    return invoke[1]!.trim();
  }
  for (const line of body.split(/\r?\n/)) {
    const t = line.trim();
    if (!t || t.startsWith("<")) {
      continue;
    }
    const ident = t.match(/^[A-Za-z][A-Za-z0-9_.-]{0,127}/);
    if (ident) {
      return ident[0];
    }
  }
  return "";
}

/**
 * Tool XML Copilot/Cursor dump into assistant content when llama.cpp does not
 * emit OpenAI `delta.tool_calls`. Supports:
 * - Qwen/Hermes: `<tool_call><function=name><parameter=k>v</parameter>`
 * - Cursor/Copilot: `<tool_call>\nname\n<arg_key>k</arg_key><arg_value>v</arg_value>`
 * Tiny models often omit closing tags; those blocks are still parsed.
 */
export function parseXmlToolCalls(text: string): XmlToolCall[] {
  const results: XmlToolCall[] = [];
  if (!text) {
    return results;
  }
  const startRe = /<tool_call\b([^>]*)>/gi;
  let start: RegExpExecArray | null;
  while ((start = startRe.exec(text)) !== null) {
    const from = start.index;
    const innerStart = start.index + start[0].length;
    const rest = text.slice(innerStart);
    const close = rest.search(/<\/tool_call>/i);
    const nextOpen = rest.search(/<tool_call\b/i);
    let innerEnd: number;
    let rawEnd: number;
    if (close >= 0 && (nextOpen < 0 || close < nextOpen)) {
      innerEnd = innerStart + close;
      rawEnd = innerEnd + "</tool_call>".length;
    } else if (nextOpen >= 0) {
      innerEnd = innerStart + nextOpen;
      rawEnd = innerEnd;
    } else {
      innerEnd = text.length;
      rawEnd = text.length;
    }
    const body = text.slice(innerStart, innerEnd);
    const name = toolNameFromOpenTag(start[1] || "") || toolNameFromBody(body);
    if (!name) {
      startRe.lastIndex = Math.max(startRe.lastIndex, rawEnd);
      continue;
    }
    results.push({
      name,
      input: parseParamTags(body),
      raw: text.slice(from, rawEnd),
    });
    startRe.lastIndex = Math.max(startRe.lastIndex, rawEnd);
  }
  return results;
}

export function stripXmlToolCalls(text: string): string {
  return text
    .replace(/<tool_call\b[\s\S]*?<\/tool_call>/gi, "")
    .replace(/<tool_call\b[\s\S]*$/gi, "")
    .trim();
}
