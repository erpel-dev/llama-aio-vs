#!/usr/bin/env python3
"""Optional pinned Strata/Maya request-parser check; requires jinja2 and compiled core.

Executes reviewed upstream frontend code, without weights, GPU or server startup.
The default npm tests use the catalog snapshot and mock HTTP endpoints offline.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = json.loads((ROOT / "packages/core/test/fixtures/remote-model-catalogs.json").read_text())


def fetch(source, path):
    target = Path(tempfile.gettempdir()) / "llama-aio-remote-contracts" / source["repo"] / source["revision"] / path
    if not target.exists():
        url = f'https://raw.githubusercontent.com/{source["repo"]}/{source["revision"]}/{path}'
        with urllib.request.urlopen(url, timeout=45) as response:
            text = response.read().decode()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return target


def main():
    code = """
      const c = require('./packages/core/out/modelModes.js');
      const data = require('./packages/core/test/fixtures/remote-model-catalogs.json');
      console.log(JSON.stringify(data.rows.map(row => ({...row,
        profile: c.resolveRemoteModelModes({modelId: row.servedId})}))));
    """
    rows = json.loads(subprocess.check_output(["node", "-e", code], cwd=ROOT))
    count = 0
    for source, provider in zip(SNAPSHOT["sources"], ["strata", "maya"]):
        path = fetch(source, "serve/frontend.py")
        spec = importlib.util.spec_from_file_location(provider + "_frontend", path)
        frontend = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = frontend
        spec.loader.exec_module(frontend)
        if provider == "maya":
            fetch(source, "serve/ds41_encoding.py")
            sys.path.insert(0, str(path.parent.parent))
            template = frontend.ChatTemplate(fetch(source, "tools/ds4/chat_template.jinja"))
        else:
            frontend.use_family("deepseek41")
            template = frontend.ChatTemplate(fetch(source, "serve/fixtures/deepseek41/chat_template.jinja"))

        for row in [row for row in rows if row["provider"] == provider]:
            deepseek = row["family"] in ["deepseek", "deepseek41"]
            if provider == "strata":
                frontend.use_family("deepseek41" if deepseek else "qwen4exp")
            for mode, params in row["profile"]["modes"].items():
                body = {"model": row["servedId"], "messages": [{"role": "user", "content": "Hello"}],
                        "chat_template_kwargs": params.get("chat_template_kwargs", {})}
                result = frontend.openai_to_messages(body, deepseek=deepseek) if provider == "maya" else frontend.openai_to_messages(body)
                messages, tools, kwargs = result
                expected = body["chat_template_kwargs"]
                assert (kwargs.get("enable_thinking") is False) == (expected.get("enable_thinking") is False), (row, mode, kwargs)
                if "reasoning_effort" in expected:
                    assert kwargs["reasoning_effort"] == expected["reasoning_effort"], (row, mode, kwargs)
                if deepseek:
                    prompt = template.render(messages, tools, **kwargs).rstrip()
                    if expected.get("enable_thinking") is False:
                        assert prompt.endswith("</think>"), (provider, mode, prompt[-100:])
                    else:
                        assert prompt.endswith("<think>"), (provider, mode, prompt[-100:])
                        assert f'Reasoning Effort: {expected["reasoning_effort"]}' in prompt
                count += 1
        # The old request field was silently dropped by both upstream parsers.
        if provider == "strata":
            frontend.use_family("deepseek41")
        old = {"messages": [{"role": "user", "content": "Hello"}], "chat_template_kwargs": {"thinking_mode": "chat"}}
        result = frontend.openai_to_messages(old, deepseek=True) if provider == "maya" else frontend.openai_to_messages(old)
        assert result[2].get("enable_thinking") is not False
        print(f'{source["repo"]}@{source["revision"][:12]}: request controls verified', flush=True)
    print(f"PASS: {count} mode requests across {len(rows)} catalog variants")


if __name__ == "__main__":
    main()
