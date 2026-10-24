# Strata and Project Maya remote profile audit

Checked October 24, 2026 against:

- [Strata 7dd5cf18e68c](https://github.com/Niko1221/Strata/tree/7dd5cf18e68c639a8773e096c5c79f971ea2a9f5): setup/model tables, Desktop labels and API routes, standalone API, and request normalization.
- [Project Maya f26d8bab43f9](https://github.com/mw00/project-maya/tree/f26d8bab43f9b10f0475986b541dc46ada45285e): installer catalogs, served names, API routes, and request normalization.

All **25 native Strata variants** and **14 Maya variants** select compatible
profiles using their normal published identities/metadata. The five entries in
Strata's legacy display catalog are covered too. The exact IDs, labels, expected
profiles, and source revisions are in the [test snapshot](../packages/core/test/fixtures/remote-model-catalogs.json).

| Source | Variants | Published identity / Desktop ID pattern | Profile |
| --- | ---: | --- | --- |
| Strata original | 4 | `qwen3.8-flash-next-*`; Desktop `q2_0`, `iq2_xs`, `iq3_xxs`, `iq3_s` | Qwen3.8 Flash-Next |
| Strata Swift | 3 | `swift-1.5-*`; Desktop `swift-*` | Qwen3.8 Flash-Next |
| Strata Coder | 1 | `qwen3.8-flash-next-coder-*`; Desktop `coder-iq1_m` | Qwen3.8 Flash-Next, not original Qwen3-Coder |
| Strata Unsloth | 5 | `qwen3.8-flash-next-unsloth-*`; Desktop `unsloth-*` | Qwen3.8 Flash-Next |
| Strata Gyro | 2 | `qwen3.8-flash-next-gyro-*`; Desktop `gyro-gyro-s`, `gyro-gyro-m` | Qwen3.8 Flash-Next |
| Strata NVFP4 | 3 | `qwen3.8-flash-next-nvfp4-*`; Desktop `nvfp4-*` | Qwen3.8 Flash-Next |
| Strata Qwen3.6 | 3 | `qwen3.6-35b-a3b-*`; Desktop `qwen36-*` | Qwen3.5 / 3.6 |
| Strata DeepSeek | 4 | `deepseek-v4.1-flash`; Desktop `deepseek-v4.1-flash-ist-das-{1.75,2.00,2.25}bpw` / `*-ud-q8_k_xl` | DeepSeek V4.1 Flash / 1017 |
| Maya GLM | 7 | `glm-5.3-flash` for all Maya-S/S24/M/L/Derisked/EXL3 installer choices | GLM-5.3-Flash |
| Maya DeepSeek | 7 | `deepseek-v4.1-flash` for all GGUF/Maya/EXL3 installer choices | DeepSeek V4.1 Flash / 1017 |
| Legacy Strata display list | 5 | Full family in `meta.label`; short IDs in the snapshot | Qwen2.5 0.5B Instruct, Qwen3-Coder Instruct, original Qwen3 |

Desktop adds instance suffixes such as `--2`; the family label still identifies
the model and the original suffixed ID is retained in chat requests. Mac and
Intel installer choices are subsets of the Strata catalog above. Preview/design
entries are not additional downloadable checkpoints. Image-generation and
draft-only models are not chat-profile candidates.

## Gaps found and fixed

1. **DeepSeek Off was ignored.** Both native request parsers discard incoming
   `thinking_mode`. They accept `enable_thinking: false`. The profile now uses
   that field, with numeric budgets 50/75/100 for Low/High/Max.
2. **Undated DeepSeek IDs selected a different sampler.** Both projects use
   `deepseek-v4.1-flash` for 1017. Top-p 1 now covers both versions; it is also a
   published option for the original checkpoint. Official IST-DAS tags resolve
   directly as well.
3. **Published aliases and fork labels were missed.** Monitor and chat now share
   parsing for `alias_of`, top-level `aliases`, `meta.aliases`, and `meta.label`.
   Actual canonical identities take precedence over alias guesses. Conflicting
   alias families produce no curated profile.
4. **A reused ID could retain stale modes.** Identity changes invalidate picker
   probes. Fresh aliases beat remembered labels, and an identified unknown or
   ambiguous model clears the old cached profile.
5. **GLM history cleanup could be discarded by the adapter.** The client applies
   `clear_thinking` to earlier assistant turns itself, retaining reasoning and
   tool calls after the latest user prompt.
6. **The legacy Qwen2.5 0.5B starter lacked a profile.** It now has the pinned
   generation recipe: temperature .7, top-p .8, top-k 20, repeat penalty 1.1,
   with no fictitious thinking modes.

## Limits and verification

An opaque ID without identifying metadata cannot be recognized reliably.
`q2_0`, `Maya-M`, `pack/full`, and generic bpw filenames are not global model
names. In particular, Maya's installer uses related Maya-M names across GLM and
DeepSeek choices. They must not be mapped by string alone. Custom imports and
future catalog additions need a recognizable canonical name, filename, label,
or an unambiguous published alias. Native engine sampler limits still apply.

`npm test` checks the complete pinned matrix and extension HTTP requests,
including Desktop instance suffixes, aliases, unchanged request IDs, selected
modes, profile switches, and cache invalidation. This is identity/protocol
coverage; the full GPU checkpoints are not loaded by this audit.

After compiling core, the optional check below executes the two pinned upstream
frontend modules and renders their DeepSeek prompts without starting an engine:

```sh
npm run compile -w @llama-aio/core
python3 -m venv /tmp/llama-aio-remote-check
/tmp/llama-aio-remote-check/bin/pip install 'jinja2==3.1.6'
/tmp/llama-aio-remote-check/bin/python scripts/verify-remote-catalogs.py
```

It verifies **146 mode requests across all 39 native variants**, including the
actual thinking/Off prompt markers and numeric DeepSeek budgets. It also
reproduces the old ignored-field behavior. The revisions are pinned for
repeatability; this is not a promise about future upstream catalogs.
