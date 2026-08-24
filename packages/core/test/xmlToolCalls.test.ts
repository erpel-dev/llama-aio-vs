import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { parseXmlToolCalls, stripXmlToolCalls } from "../src/xmlToolCalls";

describe("parseXmlToolCalls", () => {
  it("parses Qwen/Hermes function= / parameter= blocks", () => {
    const text = `<tool_call>
<function=read_file>
<parameter=target_file>
/tmp/a.ts
</parameter>
<parameter=offset>
1
</parameter>
</function>
</tool_call>`;
    const calls = parseXmlToolCalls(text);
    assert.equal(calls.length, 1);
    assert.equal(calls[0]?.name, "read_file");
    assert.deepEqual(calls[0]?.input, { target_file: "/tmp/a.ts", offset: 1 });
  });

  it("parses Cursor/Copilot arg_key / arg_value blocks", () => {
    const text = `<tool_call>
read_file
<arg_key>target_file</arg_key>
<arg_value>/home/timo/repos/llama-aio-vs/packages/tui/src/app.ts</arg_value>
<arg_key>offset</arg_key>
<arg_value>420</arg_value>
<arg_key>limit</arg_key>
<arg_value>460</arg_value>
</tool_call>`;
    const calls = parseXmlToolCalls(text);
    assert.equal(calls.length, 1);
    assert.equal(calls[0]?.name, "read_file");
    assert.deepEqual(calls[0]?.input, {
      target_file: "/home/timo/repos/llama-aio-vs/packages/tui/src/app.ts",
      offset: 420,
      limit: 460,
    });
  });

  it("parses a grep_search call the same way", () => {
    const text = `<tool_call>
grep_search
<arg_key>query</arg_key>
<arg_value>install.*download | download.*install | llama.*install</arg_value>
</tool_call>
Some leftover prose.`;
    const calls = parseXmlToolCalls(text);
    assert.equal(calls.length, 1);
    assert.equal(calls[0]?.name, "grep_search");
    assert.equal(
      (calls[0]?.input as { query: string }).query,
      "install.*download | download.*install | llama.*install"
    );
    assert.equal(stripXmlToolCalls(text), "Some leftover prose.");
  });

  it("still parses when the tiny model omits closing tags", () => {
    const text = `<tool_call>
read_file
<arg_key>target_file</arg_key>
<arg_value>/tmp/app.ts</arg_value>
<arg_key>offset</arg_key>
<arg_value>1</arg_value>`;
    const calls = parseXmlToolCalls(text);
    assert.equal(calls.length, 1);
    assert.equal(calls[0]?.name, "read_file");
    assert.equal((calls[0]?.input as { target_file: string }).target_file, "/tmp/app.ts");
    assert.equal(stripXmlToolCalls(text), "");
  });

  it("parses two consecutive calls", () => {
    const text = `<tool_call>
read_file
<arg_key>target_file</arg_key>
<arg_value>/a.ts</arg_value>
</tool_call>
<tool_call>
<function=grep_search>
<parameter=query>foo</parameter>
</function>
</tool_call>`;
    const calls = parseXmlToolCalls(text);
    assert.equal(calls.length, 2);
    assert.equal(calls[0]?.name, "read_file");
    assert.equal(calls[1]?.name, "grep_search");
  });
});
