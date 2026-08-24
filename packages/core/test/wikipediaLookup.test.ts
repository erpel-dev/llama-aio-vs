import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  lookupWikipedia,
  normalizeWikiQuery,
  wikipediaLookupToolDefinition,
  wikipediaQueryFromInput,
  WIKIPEDIA_LOOKUP_TOOL_NAME,
  type WikipediaFetch,
} from "../src/wikipediaLookup";

function jsonFetch(byUrl: (url: string) => unknown): WikipediaFetch {
  return async (url) => ({
    ok: true,
    status: 200,
    json: async () => byUrl(url),
    text: async () => JSON.stringify(byUrl(url)),
  });
}

describe("wikipediaQueryFromInput", () => {
  it("reads query and common aliases", () => {
    assert.equal(wikipediaQueryFromInput({ query: "Ada Lovelace" }), "Ada Lovelace");
    assert.equal(wikipediaQueryFromInput({ q: "Vulkan" }), "Vulkan");
    assert.equal(wikipediaQueryFromInput({ title: " RISC-V " }), "RISC-V");
  });

  it("pulls a title out of a Wikipedia URL", () => {
    assert.equal(
      normalizeWikiQuery("https://en.wikipedia.org/wiki/Ada_Lovelace"),
      "Ada Lovelace"
    );
  });
});

describe("wikipediaLookupToolDefinition", () => {
  it("exposes a one-argument function tool", () => {
    const def = wikipediaLookupToolDefinition();
    assert.equal(def.function.name, WIKIPEDIA_LOOKUP_TOOL_NAME);
    assert.deepEqual((def.function.parameters as { required?: string[] }).required, ["query"]);
  });
});

describe("lookupWikipedia", () => {
  it("returns the lead extract of the top search hit", async () => {
    const fetchImpl = jsonFetch((url) => {
      if (url.includes("list=search")) {
        return {
          query: { search: [{ title: "Ada Lovelace" }, { title: "Lovelace (crater)" }] },
        };
      }
      return {
        query: {
          pages: {
            "1": {
              title: "Ada Lovelace",
              extract: "Augusta Ada King, Countess of Lovelace, was an English mathematician.",
              fullurl: "https://en.wikipedia.org/wiki/Ada_Lovelace",
            },
          },
        },
      };
    });
    const text = await lookupWikipedia("ada lovelace", { fetch: fetchImpl });
    assert.match(text, /^Wikipedia: Ada Lovelace/);
    assert.match(text, /Countess of Lovelace/);
    assert.match(text, /Also matched: Lovelace \(crater\)/);
    assert.match(text, /en\.wikipedia\.org\/wiki\/Ada_Lovelace/);
  });

  it("says so when search is empty", async () => {
    const fetchImpl = jsonFetch(() => ({ query: { search: [] } }));
    const text = await lookupWikipedia("zzzz-not-an-article", { fetch: fetchImpl });
    assert.match(text, /No English Wikipedia article matched/);
  });

  it("rejects a blank query", async () => {
    const text = await lookupWikipedia("  ");
    assert.match(text, /non-empty query/);
  });
});
