/** Copilot / llama.cpp function name. Keep it short for tiny models. */
export const WIKIPEDIA_LOOKUP_TOOL_NAME = "wikipedia_lookup";

export const WIKIPEDIA_LOOKUP_TOOL_DESCRIPTION =
  "Look up a topic on English Wikipedia and return the lead section. Use this for facts, people, history, science, and general knowledge a small model may not know. Pass a search query, not a URL. Do not use this for source code, APIs, or files in the workspace.";

export const WIKIPEDIA_LOOKUP_SYSTEM_HINT =
  "If a fact is not in the workspace and you are unsure, call wikipedia_lookup with a short query before answering. Do not guess encyclopedic facts. Do not use Wikipedia for APIs, libraries, or source code — grep the repo instead.";

const WIKI_API = "https://en.wikipedia.org/w/api.php";
const USER_AGENT =
  "llama-aio-vs/0.1.14 (https://github.com/erpel-dev/llama-aio-vs; Wikipedia lookup for local LLMs)";
const EXTRACT_CHARS = 1800;

export type WikipediaFetch = (
  url: string,
  init?: { headers?: Record<string, string>; signal?: AbortSignal }
) => Promise<{ ok: boolean; status: number; json: () => Promise<unknown>; text: () => Promise<string> }>;

export function wikipediaLookupToolDefinition(): {
  type: "function";
  function: { name: string; description: string; parameters: object };
} {
  return {
    type: "function",
    function: {
      name: WIKIPEDIA_LOOKUP_TOOL_NAME,
      description: WIKIPEDIA_LOOKUP_TOOL_DESCRIPTION,
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Search query or article title, e.g. 'Ada Lovelace' or 'Vulkan graphics API'",
          },
        },
        required: ["query"],
      },
    },
  };
}

/** Tiny models often put the topic in q/search/title instead of query. */
export function wikipediaQueryFromInput(input: unknown): string {
  if (typeof input === "string") {
    return normalizeWikiQuery(input);
  }
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    return "";
  }
  const obj = input as Record<string, unknown>;
  for (const key of ["query", "q", "search", "title", "topic"]) {
    const v = obj[key];
    if (typeof v === "string" && v.trim()) {
      return normalizeWikiQuery(v);
    }
  }
  return "";
}

export function normalizeWikiQuery(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) {
    return "";
  }
  const wikiUrl = /wikipedia\.org\/wiki\/([^?#]+)/i.exec(trimmed);
  if (wikiUrl) {
    try {
      return decodeURIComponent(wikiUrl[1]!).replace(/_/g, " ").trim();
    } catch {
      return wikiUrl[1]!.replace(/_/g, " ").trim();
    }
  }
  return trimmed;
}

function truncateText(text: string, maxChars: number): string {
  const t = text.replace(/\s+/g, " ").trim();
  if (t.length <= maxChars) {
    return t;
  }
  const slice = t.slice(0, maxChars);
  const sp = slice.lastIndexOf(" ");
  return `${(sp > maxChars * 0.6 ? slice.slice(0, sp) : slice).trimEnd()}…`;
}

async function wikiJson(
  url: string,
  fetchImpl: WikipediaFetch,
  signal?: AbortSignal
): Promise<unknown> {
  const res = await fetchImpl(url, {
    headers: { "User-Agent": USER_AGENT, Accept: "application/json" },
    signal,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Wikipedia HTTP ${res.status}${body ? `: ${body.slice(0, 180)}` : ""}`);
  }
  return res.json();
}

function searchTitles(data: unknown): string[] {
  const hits = (data as { query?: { search?: Array<{ title?: string }> } })?.query?.search;
  if (!Array.isArray(hits)) {
    return [];
  }
  return hits.map((h) => (h.title || "").trim()).filter(Boolean);
}

function pageExtract(data: unknown): { title: string; extract: string; url: string } | undefined {
  const pages = (data as { query?: { pages?: Record<string, unknown> } })?.query?.pages;
  if (!pages || typeof pages !== "object") {
    return undefined;
  }
  for (const page of Object.values(pages)) {
    if (!page || typeof page !== "object") {
      continue;
    }
    const p = page as {
      missing?: boolean;
      title?: string;
      extract?: string;
      fullurl?: string;
    };
    if (p.missing) {
      continue;
    }
    const title = (p.title || "").trim();
    const extract = (p.extract || "").trim();
    if (!title || !extract) {
      continue;
    }
    return {
      title,
      extract,
      url: typeof p.fullurl === "string" ? p.fullurl : `https://en.wikipedia.org/wiki/${encodeURIComponent(title.replace(/ /g, "_"))}`,
    };
  }
  return undefined;
}

export async function lookupWikipedia(
  query: string,
  options?: { fetch?: WikipediaFetch; signal?: AbortSignal }
): Promise<string> {
  const q = wikipediaQueryFromInput(query) || query.trim();
  if (!q) {
    return "Wikipedia lookup needs a non-empty query.";
  }
  const fetchImpl = options?.fetch ?? (globalThis.fetch as WikipediaFetch);
  if (typeof fetchImpl !== "function") {
    return "Wikipedia lookup is unavailable (no fetch).";
  }

  const searchUrl =
    `${WIKI_API}?action=query&list=search&srsearch=${encodeURIComponent(q)}` +
    `&srlimit=3&srprop=&format=json`;
  let titles: string[] = [];
  try {
    titles = searchTitles(await wikiJson(searchUrl, fetchImpl, options?.signal));
  } catch (err) {
    return `Wikipedia search failed: ${err instanceof Error ? err.message : String(err)}`;
  }
  if (!titles.length) {
    return `No English Wikipedia article matched “${q}”.`;
  }

  const title = titles[0]!;
  const extractUrl =
    `${WIKI_API}?action=query&prop=extracts|info&exintro=1&explaintext=1&redirects=1` +
    `&inprop=url&titles=${encodeURIComponent(title)}&format=json`;
  let page: { title: string; extract: string; url: string } | undefined;
  try {
    page = pageExtract(await wikiJson(extractUrl, fetchImpl, options?.signal));
  } catch (err) {
    return `Wikipedia extract failed: ${err instanceof Error ? err.message : String(err)}`;
  }
  if (!page) {
    return `Wikipedia had a hit for “${q}” (${title}) but no extract.`;
  }

  const others = titles.filter((t) => t !== page.title).slice(0, 2);
  const lines = [
    `Wikipedia: ${page.title}`,
    page.url,
    truncateText(page.extract, EXTRACT_CHARS),
  ];
  if (others.length) {
    lines.push(`Also matched: ${others.join("; ")}`);
  }
  return lines.join("\n");
}
