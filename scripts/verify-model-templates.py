#!/usr/bin/env python3
"""Optional network check: pip install jinja2; compile core, then run this script.

Renders every advertised mode against pinned author templates, including tool
history. This checks prompt construction, not inference quality or GPU support.
Use --cache DIR to reuse downloaded HF files (ID directories use __ for /).
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import urllib.request

from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class Generation(Extension):
    """HF's generation tag marks assistant spans; rendering is unchanged."""
    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return nodes.Scope(parser.parse_statements(["name:endgeneration"], drop_needle=True))


def reject(message):
    raise ValueError(message)


def gguf_template(url):
    """Read metadata only; close after 32 MiB even if Range is ignored."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-33554431"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(33554432)
    offset = 0

    def take(fmt):
        nonlocal offset
        value = struct.unpack_from("<" + fmt, data, offset)[0]
        offset += struct.calcsize("<" + fmt)
        return value

    def string():
        nonlocal offset
        size = take("Q")
        if offset + size > len(data):
            raise ValueError("GGUF metadata exceeds the bounded prefix")
        value = data[offset:offset + size]
        offset += size
        return value.decode()

    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            item, count = take("I"), take("Q")
            for _ in range(count):
                value(item)
            return None
        return take({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                     7: "?", 10: "Q", 11: "q", 12: "d"}[kind])

    assert take("I") == 0x46554747
    assert take("I") in [2, 3]
    take("Q")  # tensor count
    for _ in range(take("Q")):
        key = string()
        result = value(take("I"))
        if key == "tokenizer.chat_template":
            return result
    raise ValueError("No tokenizer.chat_template in GGUF metadata")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/llama-aio-model-templates"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    sources = json.loads((root / "scripts/model-profile-sources.json").read_text())
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                       extensions=[Generation, "jinja2.ext.loopcontrols"])
    env.globals.update(raise_exception=reject, strftime_now=lambda fmt: datetime.datetime(2026, 10, 24).strftime(fmt))
    env.filters["tojson"] = lambda value, **kw: json.dumps(value, **{"ensure_ascii": False, **kw})
    tool = {"type": "function", "function": {"name": "read_file", "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
    plain = [{"role": "user", "content": "Read README.md."}]
    history = plain + [{"role": "assistant", "content": "", "reasoning_content": "I need the file.",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "read_file", "arguments": {"path": "README.md"}}}]},
        {"role": "tool", "name": "read_file", "tool_call_id": "call_1", "content": "Example project."},
        {"role": "user", "content": "Summarize it."}]
    count = 0
    for source in sources:
        folder = args.cache / source["id"].replace("/", "__") / source["revision"]
        folder.mkdir(parents=True, exist_ok=True)

        def fetch(name, optional=False):
            target = folder / name
            if target.exists():
                return target.read_text()
            url = f'https://huggingface.co/{source["id"]}/raw/{source["revision"]}/{name}'
            try:
                with urllib.request.urlopen(url, timeout=45) as response:
                    value = response.read().decode()
            except urllib.error.HTTPError as error:
                if optional and error.code == 404:
                    return None
                raise
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(value)
            return value

        tokenizer = {}
        encoder = None
        if "gguf" in source:
            cached = folder / "extracted-chat-template.jinja"
            if not cached.exists():
                url = f'https://huggingface.co/{source["id"]}/resolve/{source["revision"]}/{source["gguf"]}'
                cached.write_text(gguf_template(url))
            template = cached.read_text()
        elif "encoder" in source:
            # Optional, pinned author reference encoder (reviewed stdlib-only
            # source); this check is intentionally outside the default test run.
            template = fetch(source["encoder"])
            namespace = {"__name__": "author_reference_encoder"}
            exec(compile(template, source["encoder"], "exec"), namespace)
            encoder = namespace["encode_messages"]
        else:
            tokenizer = json.loads(fetch("tokenizer_config.json"))
            template = fetch("chat_template.jinja", optional=True) or tokenizer["chat_template"]
            if isinstance(template, list):
                template = next(item["template"] for item in template if item["name"] == "default")
        compiled = env.from_string(template) if encoder is None else None
        code = "console.log(JSON.stringify(require('./packages/core/out/modelModes.js').resolveModelModes(undefined, process.argv[1])))"
        profile = json.loads(subprocess.check_output(["node", "-e", code, source["id"]], cwd=root))
        outputs = {}
        for mode, params in profile["modes"].items():
            for fixture, messages in [("plain", plain), ("tools", history)]:
                context = {key: (value.get("content", "") if isinstance(value, dict) else value)
                           for key, value in tokenizer.items() if key.endswith("_token")}
                context.update(messages=messages, tools=[tool], add_generation_prompt=True)
                context.update(params.get("chat_template_kwargs", {}))
                if encoder:
                    conversation = copy.deepcopy(messages)
                    conversation.insert(0, {"role": "system", "content": "", "tools": [tool]})
                    for message in conversation:
                        for call in message.get("tool_calls", []):
                            call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
                    kwargs = dict(params.get("chat_template_kwargs", {}))
                    if "enable_thinking" in kwargs:
                        kwargs["thinking_mode"] = "thinking" if kwargs.pop("enable_thinking") else "chat"
                    rendered = encoder(conversation, **kwargs)
                else:
                    rendered = compiled.render(**context)
                assert rendered.strip(), (source["id"], mode, fixture)
                if fixture == "tools":
                    assert "read_file" in rendered and "Example project." in rendered
                outputs[mode, fixture] = rendered
                count += 1
        # Sampler-only presets can share a prompt. Distinct template controls
        # must actually change the prompt, otherwise the picker has no effect.
        modes = list(profile["modes"].items())
        for index, (left, lp) in enumerate(modes):
            for right, rp in modes[index + 1:]:
                if lp.get("chat_template_kwargs") != rp.get("chat_template_kwargs"):
                    assert outputs[left, "plain"] != outputs[right, "plain"], (source["id"], left, right)
        digest = hashlib.sha256(template.encode()).hexdigest()[:12]
        print(f'{source["id"]}@{source["revision"][:12]} template={digest}: {len(modes)} modes OK', flush=True)
    print(f"PASS: {count} template renders across {len(sources)} checkpoints")


if __name__ == "__main__":
    main()
