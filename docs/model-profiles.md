# Model request profiles

Profiles select sampling parameters and chat-template controls in Copilot Chat
and Copilot CLI. They do not install a model runtime or change its served context
size. Your server must already support the model, its GGUF architecture, its
chat template, and its reasoning/tool-call format.

The picker exposes only the controls supported by each verified model version.
CLI effort values select the same presets as VS Code labels; saved labels from
older extension versions remain accepted where that mode is still supported. A missing or unsupported selection
uses that profile's default. General/Coding are sampling recipes, not claims
that the model has different reasoning effort levels. Nanbeige's Agent/Chat
presets are likewise scenario recipes, not an effort ladder.

| Checkpoint family | Available modes (default in bold) | Sampling / distinctions |
| --- | --- | --- |
| Spark X2.5 | **Think**, No Think | Temperature 1, top-p .95, no top-k filtering |
| Spark X2.6 | **High**, Low, No Think | Same sampler; only high/low reasoning efforts are valid |
| GPT-OSS 20B / 120B | High, **Medium**, Low | Temperature 1, unfiltered top-p/k; requires Harmony support |
| GLM-5.3-Flash | **Max**, High, Low | Temperature 1, top-p .95; clear old thinking on each turn |
| GLM-5.3 full | **Max**, High, Low | Same controls and sampler as Flash; requires a compatible large-model server |
| DeepSeek V4.1 Flash / 1017 | Max, **High**, Low, No Think | Temperature 1, top-p 1; numeric budgets 100/75/50 |
| Tiel-Coder 35B-A3B | **Think**, No Think | Coding recipe .6/.95/20; off follows the Qwen base's .7/.8/20, presence 1.5 |
| MiniCPM5 2B | **Think**, No Think | 1/.95, min-p 0; no top-k filtering |
| Nanbeige4.3 33B | **Agent**, Chat, No Think | Agent: 1/.95, preserve thinking. Chat/off: .6/.95, clear earlier thinking. XML tools; no top-k filtering |
| Mellum2.1 12B-A2.5B Thinking | **Think** only | Author's quickstart recipe .6/.95/20 |
| MiMo V2.6 Distill Qwen 9B | **Think** only | .6/.95/20; no invented effort ladder |
| Underdog Saluki 27B 1.0 | **Think**, No Think | Think .6/.95/20; direct tool calls use temperature 0 |
| Flint | **Think** only | 1/.95/20; preserve returned reasoning across tool turns |
| Sage A1.5 | **Think** only | 1/.95/20, presence 1.5, min-p 0; standard agent recipe |
| Ornith 1.5 9B / 35B-A3B / 122B | General, **Coding**, No Think | Verified Qwen-based variants; same recipes as Qwen3.5/3.6 |
| Nemotron 3 Nano 4B / 30B-A3B | General, **Coding**, No Think | General: temperature 1, top-p .95 (4B) or 1 (30B). Coding/off: .6/.95 |
| Nemotron 3.5 Bolt E4B | **Think**, No Think | Gemma-based, temperature 1, top-p .95, top-k 64 |
| LFM2.5 230M / 1.2B Instruct | **No Think** only | Temperature .1, top-k 50, repeat penalty 1.05 |
| LFM2.5 2.6B | **Think** only | Temperature .1, top-k 50, repeat penalty 1.1 |
| LFM2.5 24B-A2B | **Think** only | Temperature .1, top-k 50, repeat penalty 1.05 |
| Ternary Bonsai 2 27B | **XHigh**, Medium, No Think | Thinking min-p .05; no Low mode; requires Prism's llama.cpp fork |
| Qwen3 original | **Think**, No Think | Think: .6/.95/20; off: .7/.8/20; presence penalty 0 |
| Qwen2.5 0.5B Instruct (legacy Strata starter) | **No Think** only | .7/.8/20, repeat penalty 1.1 |
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

Tiel uses the coding sampler without overriding its embedded template's default
reasoning effort. The MTP checkpoint contains a custom template; it is checked
directly from the GGUF header rather than assumed identical to the base.
Saluki likewise uses its own embedded template. Mellum's published quickstart
uses temperature .6 although its agent benchmark description uses 1; the
profile deliberately follows the quickstart. Sage's specialized verification
protocol is not enabled by a normal chat preset.

Nanbeige4.3 matches only the verified 33B chat checkpoint, including published
GGUF names such as `Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf`. Base, 4.2, future
3B/4.4 versions, and prediction sidecars do not inherit this profile. Agent is
the default recipe for coding/tool use and preserves prior reasoning. Chat
clears reasoning from earlier user turns while retaining the current tool
turn. No Think uses the chat sampler with `enable_thinking: false`; the author
does not prescribe a separate off-mode sampler. `tool_call_format: xml` follows
the publisher's recommendation. Undocumented top-k/min-p filters and penalties
are neutralized. The profile keeps the user's context and output-token limits.

Copilot SDK/CLI effort values `high`, `medium`, and `none` select Nanbeige Agent,
Chat, and No Think respectively; these values select presets rather than setting
an unsupported `reasoning_effort` template parameter.

DeepSeek requires a V4.1-compatible engine that forwards `enable_thinking` and
numeric `reasoning_effort` from `chat_template_kwargs` into its template or
reference encoder and parses DSML tool calls into structured OpenAI tool calls.
The original checkpoint has a Jinja template; 1017 is checked with the pinned
author reference encoder. Hosted APIs with different parameter conventions may
need a server-side adapter. Flint requires a llama.cpp build including its
October 21 architecture support (or another compatible engine).

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

Published `alias_of`, `aliases`, and `meta.aliases` can identify an opaque model
too; conflicting alias families leave sampling free. Human-readable `meta.label`
is accepted from compatible forks as well as Strata. Alias changes invalidate
picker probes so a reused served ID cannot keep another model's profile.
See the [Strata/Maya catalog audit](remote-catalog-audit.md) for the complete
pinned variant matrix and request-parser verification.

Strata and Maya publish an undated DeepSeek ID for the 1017 checkpoint. Top-p 1
is supported by both cards and matches both native servers. They ignore the
incoming `thinking_mode` field, so No Think uses `enable_thinking: false` instead.
GLM's old-reasoning cleanup is also applied on the client, since these adapters
can discard `clear_thinking`. The client likewise clears earlier reasoning when
a profile sends `preserve_thinking: false`, including Nanbeige Chat and No Think;
reasoning within the current tool turn is retained.

## Copilot SDK harness

When the VS Code Copilot SDK harness uses the registered Llama AIO model, its
Responses proxy and renderer bridge forward `reasoning.effort` as the Language
Model API's model configuration. The same standard effort values and curated
presets therefore apply to local and remote servers, including No Think (`none`).
The backend still receives Chat Completions with template controls nested in
`chat_template_kwargs`; it does not need a Responses endpoint. Missing effort
selections use the advertised profile default.

Nanbeige4.3, Flint, and DeepSeek reasoning is streamed as separate thinking parts on newer
hosts and sent back as assistant `reasoning_content` on later tool turns.
It is counted in context estimates and kept out of the final answer text.
Older hosts without thinking-part support keep the existing text fallback;
use a recent VS Code host for reasoning-preserving agent sessions.
Consecutive streamed tool turns use fresh HTTP connections to avoid reusing
SSE sockets that llama-server is closing; chat POSTs are not retried.
The SSE `[DONE]` marker completes a reply even if the HTTP connection stays
open. Inline `<think>` blocks stream into the thinking channel as they arrive.
A completed thinking-only reply reports a useful error instead of succeeding
without an answer; an output-limit error points to Chat Requests → Max tokens.
View last response includes partial output while a request is running and after
cancellation or failure, along with the finish reason and thinking character count.

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

- [Nanbeige4.3 33B, pinned](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/blob/c1093c501cc6460427c0ac634db7223d0450c0ee/README.md): agent/chat samplers, binary thinking control, reasoning-history policy, XML tools, and 512K model context.
- [Bonsai 2 27B, pinned](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/b072e1d3b35a0a630cece372c2127528e0994386/README.md): sampler and supported efforts; the repository publishes GGUFs, so template verification uses its Qwen3.8 base rather than claiming a direct Bonsai template render.
- [Muse Glimmer, pinned](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/blob/b44f1072cf75e918a1b82a3a17b45c4389b076e7/README.md): existing reasoning_strength controls and sampler; no standalone author template was available for this check.

`npm test` checks identity boundaries, advertised modes, CLI selections, numeric
samplers, and complete extension HTTP requests through opaque remote IDs.
The optional check below renders every advertised mode against pinned author
templates/reference encoders with both plain chat and tool-call/result history. It also verifies
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
MiniCPM5 2B Q4_K_M also passes CPU inference on b11729 with thinking on/off and
a native structured tool call. Nanbeige4.3's pinned template passes native b11729
`/apply-template` checks for all three presets, XML tool history, and reasoning
retention (an existing MiniCPM model supplies server startup; no Nanbeige weight
inference is claimed). Other profiles have contract/request/template coverage, not a claim that every
checkpoint was loaded on every backend. Use a recent compatible server; Bonsai's
ternary weights specifically require Prism's fork, which the standard installer
does not supply. DeepSeek's 1017 encoder check executes pinned, reviewed author
Python code; it is optional and outside the default test suite. GGUF-only
template checks read a bounded 32 MiB prefix of the pinned model file instead
of downloading the full checkpoint. For hard Flint tasks, the author recommends
an output budget of at least 32K tokens; adjust Max tokens and the served
context accordingly.
