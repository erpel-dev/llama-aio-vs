# Model request profiles

Profiles select sampling parameters and chat-template controls in Copilot Chat
and Copilot CLI. They do not install a model runtime or change its served context
size. Your server must already support the model, its GGUF architecture, its
chat template, and its reasoning/tool-call format.

The picker exposes only the controls supported by each verified model version.
CLI effort values select the same presets as VS Code labels; saved labels from
older extension versions remain accepted where that mode is still supported. A missing or unsupported selection
uses that profile's default. General/Coding are sampling recipes, not claims
that the model has different reasoning effort levels.

| Checkpoint family | Available modes (default in bold) | Sampling / distinctions |
| --- | --- | --- |
| Spark X2.5 | **Think**, No Think | Temperature 1, top-p .95, no top-k filtering |
| Spark X2.6 | **High**, Low, No Think | Same sampler; only high/low reasoning efforts are valid |
| GPT-OSS 20B / 120B | High, **Medium**, Low | Temperature 1, unfiltered top-p/k; requires Harmony support |
| GLM-5.3-Flash | **Max**, High, Low | Temperature 1, top-p .95; clear old thinking on each turn |
| Ornith 1.5 9B / 35B-A3B / 122B | General, **Coding**, No Think | Verified Qwen-based variants; same recipes as Qwen3.5/3.6 |
| Nemotron 3 Nano 4B / 30B-A3B | General, **Coding**, No Think | General: temperature 1, top-p .95 (4B) or 1 (30B). Coding/off: .6/.95 |
| Nemotron 3.5 Bolt E4B | **Think**, No Think | Gemma-based, temperature 1, top-p .95, top-k 64 |
| LFM2.5 230M / 1.2B Instruct | **No Think** only | Temperature .1, top-k 50, repeat penalty 1.05 |
| LFM2.5 2.6B | **Think** only | Temperature .1, top-k 50, repeat penalty 1.1 |
| LFM2.5 24B-A2B | **Think** only | Temperature .1, top-k 50, repeat penalty 1.05 |
| Ternary Bonsai 2 27B | **XHigh**, Medium, No Think | Thinking min-p .05; no Low mode; requires Prism's llama.cpp fork |
| Qwen3 original | **Think**, No Think | Think: .6/.95/20; off: .7/.8/20; presence penalty 0 |
| Qwen3 Instruct 2507 | **No Think** only | .7/.8/20 |
| Qwen3 Thinking 2507 | **Think** only | .6/.95/20 |
| Empero Qwen3.8 35B-A3B Distill | **Think** only | .6/.95/20; Qwen3.6-based template, no effort ladder |
| Qwen3-Coder 30B-A3B / 480B-A35B Instruct | **No Think** only | .7/.8/20, repeat penalty 1.05 |
| Qwen3-Coder-Next | **No Think** only | 1/.95/40 |
| Qwen3.5 / 3.6 | General, **Coding**, No Think | General 1/.95/20, presence 1.5; Coding .6/.95/20, presence 0; off .7/.8/20, presence 1.5 |
| Qwen3.8 / Flash-Next / Swift 1.5 | **XHigh**, Medium, Low, No Think | Think 1/.95/20, presence 0; off .7/.8/20, presence 1.5 |
| Gemma 4 | **Think**, No Think | 1/.95/64 |
| Muse Glimmer | XHigh, **High**, Medium, Low | 1/.95/64; uses reasoning_strength; always thinks |

Slash-separated numeric recipes above mean temperature / top-p / top-k.
Profiles reset frequency penalty to 0 and repeat penalty to 1 unless specified.
They disable min-p unless specified, and disable top-k when no cutoff is given.
Unspecified presence penalties are 0. This prevents unrelated request settings
or server sampler defaults from silently changing a selected recipe. GPT-OSS
has no numeric sampler prescribed in its generation config: its unfiltered
recipe is an extension choice. Nemotron's No Think preset uses the tool-calling
sampler; Ornith's No Think sampler follows its Qwen base recipe.

Model names, GGUF filenames, and server-provided canonical aliases/labels select
profiles. Opaque Strata IDs work when their metadata identifies the checkpoint;
the original served ID is retained in requests. Parent directory names do not
identify a checkpoint. Draft models, MTP sidecars, explicit Base checkpoints,
and unverified named variants are excluded. Architecture fallback is used only
where it identifies an established profile; Spark's shared architecture cannot
distinguish X2.5 from X2.6, so a versioned name takes precedence.

Remote connections use the same alias → filename → model label → served ID
lookup in chat and the sidebar, including Windows paths when the extension runs
on another OS. Local GGUF selections do not determine a remote model's profile.
The sidebar shows the default preset; each chat request uses the choice made
in its model picker. Profile-owned sampling controls are hidden because the
preset overrides them. Main models advertising embedded MTP heads remain valid;
prediction sidecars are excluded. Explicit dense Swift checkpoint names take
precedence over Strata's shortened Swift alias for Flash-Next.

## Copilot SDK harness

When the VS Code Copilot SDK harness uses the registered Llama AIO model, its
Responses proxy and renderer bridge forward `reasoning.effort` as the Language
Model API's model configuration. The same standard effort values and curated
presets therefore apply to local and remote servers, including No Think (`none`).
The backend still receives Chat Completions with template controls nested in
`chat_template_kwargs`; it does not need a Responses endpoint. Missing effort
selections use the advertised profile default.

This applies through VS Code's Llama AIO provider. A standalone SDK configured
to contact a server directly bypasses the extension: configure that client's
sampling and template parameters separately. Remote servers also need to accept
the template controls; an OpenAI-compatible API alone does not guarantee that.

After compiling core, run the optional upstream bridge contract check:

```sh
NODE_USE_ENV_PROXY=1 node scripts/verify-copilot-sdk-bridge.cjs
```

It executes pinned VS Code Responses translation and renderer bridge code,
checking profile enumeration, context limits, effort/default forwarding, and
tool history/results across all profiles. VS Code services are stubbed; this
does not launch a signed-in SDK harness session. Extension HTTP tests separately
check authenticated remote connections through a URL prefix, served model IDs,
tool conversion, and the selected sampler/template parameters.

## Sources and verification

The pinned author revisions for recipes and templates are recorded in
[model-profile-sources.json](../scripts/model-profile-sources.json). This includes
the official Qwen, XHToken, OpenAI, Z.ai, Ornith, NVIDIA, LiquidAI, and Google
model repositories. Each entry can be inspected at
`https://huggingface.co/<id>/tree/<revision>`.

Additional author cards:

- [Bonsai 2 27B, pinned](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/b072e1d3b35a0a630cece372c2127528e0994386/README.md): sampler and supported efforts; the repository publishes GGUFs, so template verification uses its Qwen3.8 base rather than claiming a direct Bonsai template render.
- [Muse Glimmer, pinned](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/blob/b44f1072cf75e918a1b82a3a17b45c4389b076e7/README.md): existing reasoning_strength controls and sampler; no standalone author template was available for this check.

`npm test` checks identity boundaries, advertised modes, CLI selections, numeric
samplers, and complete extension HTTP requests through opaque remote IDs.
The optional check below renders every advertised mode against pinned author
templates with both plain chat and tool-call/result history. It also verifies
that different template controls actually change the rendered prompt.

```sh
npm run compile -w @llama-aio/core
python3 -m venv /tmp/llama-aio-template-check
/tmp/llama-aio-template-check/bin/pip install 'jinja2==3.1.6'
/tmp/llama-aio-template-check/bin/python scripts/verify-model-templates.py
```

Validation also includes real CPU inference with Spark X2.6 4B Q4_K_M on
llama.cpp b11729: High, Low, and No Think, with plain chat and tool history.
These are short output-budget smoke checks, not quality or throughput benchmarks.
Other profiles have contract/request/template coverage, not a claim that every
checkpoint was loaded on every backend. Use a recent compatible server; Bonsai's
ternary weights specifically require Prism's fork, which the standard installer
does not supply. Flint and DeepSeek profiles remain deferred pending separate
runtime/protocol verification.
