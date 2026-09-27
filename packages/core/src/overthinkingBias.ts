/**
 * Request-time logit penalty on hesitation / backtracking words.
 *
 * The word list is the one measured in the Qwen3.8 test run (Lotfi et al.,
 * arXiv:2606.00206). Ids are not portable across tokenizers, so they are
 * resolved from the running llama-server and cached per endpoint + model.
 * A leading-space piece (" Wait") and a line-start piece ("Wait") are
 * different ids; both are kept when each string is a single token. A bare
 * word that splits (for example "re" + "consider") is skipped so the penalty
 * does not hit the first piece on its own.
 */
import * as http from "http";
import * as https from "https";

/** Words, without a leading space. Each is also tokenized with a leading space. */
export const OVERTHINKING_MARKERS = [
  "perhaps",
  "maybe",
  "wait",
  "Wait",
  "actually",
  "hold",
  "Hmm",
  "hmm",
  "Alternatively",
  "alternatively",
  "However",
  "however",
  "instead",
  "Instead",
  "But",
  "but",
  "though",
  "although",
  "yet",
  "rather",
  "unless",
  "otherwise",
  "nonetheless",
  "nevertheless",
  "regardless",
  "still",
  "anyway",
  "Or",
  "or",
  "either",
  "whether",
  "uncertain",
  "unsure",
  "possibly",
  "might",
  "could",
  "another",
  "different",
  "reconsider",
  "rethink",
  "backtrack",
  "retry",
  "revisit",
  "doubt",
  "confused",
  "wrong",
  "mistake",
  "error",
  "incorrect",
] as const;

export interface TokenizerToken {
  id: number;
}

/** One id when `tokens` is exactly one vocabulary piece; otherwise skip. */
export function singleTokenId(tokens: TokenizerToken[]): number | undefined {
  if (tokens.length !== 1) {
    return undefined;
  }
  const id = tokens[0]?.id;
  if (!Number.isInteger(id) || id < 0) {
    return undefined;
  }
  return id;
}

export function mergeTokenIds(ids: Array<number | undefined>): number[] {
  const seen = new Set<number>();
  const out: number[] = [];
  for (const id of ids) {
    if (id === undefined || seen.has(id)) {
      continue;
    }
    seen.add(id);
    out.push(id);
  }
  return out;
}

/** llama.cpp `logit_bias` map: token-id string → negative strength. */
export function logitBiasMap(ids: number[], strength: number): Record<string, number> {
  const bias = -Math.abs(strength);
  const out: Record<string, number> = {};
  for (const id of ids) {
    out[String(id)] = bias;
  }
  return out;
}

export function markerTexts(): string[] {
  const texts: string[] = [];
  for (const word of OVERTHINKING_MARKERS) {
    texts.push(` ${word}`, word);
  }
  return texts;
}

const cache = new Map<string, number[]>();
const inflight = new Map<string, Promise<number[]>>();

export function clearOverthinkingTokenCache(): void {
  cache.clear();
  inflight.clear();
}

function cacheKey(endpoint: string, modelKey: string): string {
  return `${endpoint.replace(/\/$/, "")}\n${modelKey}`;
}

async function resolveIds(tokenize: (text: string) => Promise<TokenizerToken[]>): Promise<number[]> {
  const texts = markerTexts();
  const ids: Array<number | undefined> = new Array(texts.length);
  let cursor = 0;
  async function worker(): Promise<void> {
    for (;;) {
      const i = cursor++;
      if (i >= texts.length) {
        return;
      }
      ids[i] = singleTokenId(await tokenize(texts[i]!));
    }
  }
  const workers = Math.min(8, texts.length);
  await Promise.all(Array.from({ length: workers }, () => worker()));
  return mergeTokenIds(ids);
}

/**
 * Token ids for the marker list. Cached until {@link clearOverthinkingTokenCache}.
 * `tokenize` defaults to POST `{endpoint}/tokenize`.
 */
export async function cachedOverthinkingTokenIds(
  endpoint: string,
  modelKey: string,
  tokenize?: (text: string) => Promise<TokenizerToken[]>
): Promise<number[]> {
  const key = cacheKey(endpoint, modelKey);
  const hit = cache.get(key);
  if (hit) {
    return hit;
  }
  const pending = inflight.get(key);
  if (pending) {
    return pending;
  }
  const run = resolveIds(tokenize ?? ((text) => tokenizePieces(endpoint, text)))
    .then((ids) => {
      cache.set(key, ids);
      inflight.delete(key);
      return ids;
    })
    .catch((err) => {
      inflight.delete(key);
      throw err;
    });
  inflight.set(key, run);
  return run;
}

/**
 * Bias map for a chat request, or undefined when the penalty is off or the
 * tokenizer could not be reached. A failure must not fail the chat turn.
 */
export async function overthinkingLogitBias(
  endpoint: string,
  modelKey: string,
  enabled: boolean,
  strength: number
): Promise<Record<string, number> | undefined> {
  if (!enabled || !endpoint) {
    return undefined;
  }
  try {
    const ids = await cachedOverthinkingTokenIds(endpoint, modelKey);
    if (!ids.length) {
      return undefined;
    }
    return logitBiasMap(ids, strength);
  } catch {
    return undefined;
  }
}

function postJson(url: string, body: unknown, timeoutMs: number): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const payload = JSON.stringify(body);
    const lib = u.protocol === "https:" ? https : http;
    const req = lib.request(
      {
        hostname: u.hostname,
        port: u.port,
        path: `${u.pathname}${u.search}`,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(payload),
        },
        timeout: timeoutMs,
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (c: Buffer) => chunks.push(c));
        res.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          if (!res.statusCode || res.statusCode >= 400) {
            reject(new Error(`tokenize HTTP ${res.statusCode || 0}: ${text.slice(0, 200)}`));
            return;
          }
          try {
            resolve(JSON.parse(text) as unknown);
          } catch (err) {
            reject(err);
          }
        });
      }
    );
    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy(new Error("tokenize timed out"));
    });
    req.end(payload);
  });
}

/** POST /tokenize. Accepts `{id, piece}` objects or bare id numbers. */
export async function tokenizePieces(endpoint: string, text: string): Promise<TokenizerToken[]> {
  const data = (await postJson(
    `${endpoint.replace(/\/$/, "")}/tokenize`,
    {
      content: text,
      add_special: false,
      parse_special: false,
      with_pieces: true,
    },
    8000
  )) as { tokens?: unknown };
  if (!Array.isArray(data.tokens)) {
    return [];
  }
  const out: TokenizerToken[] = [];
  for (const token of data.tokens) {
    if (typeof token === "number" && Number.isInteger(token) && token >= 0) {
      out.push({ id: token });
      continue;
    }
    if (token && typeof token === "object" && typeof (token as { id?: unknown }).id === "number") {
      const id = (token as { id: number }).id;
      if (Number.isInteger(id) && id >= 0) {
        out.push({ id });
      }
    }
  }
  return out;
}
