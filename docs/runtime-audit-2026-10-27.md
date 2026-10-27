# Extension runtime audit, October 27, 2026

The audit uses code-server 4.141.0 / VS Code 1.141.0, Chromium, the actual
VS Code Language Model API, and llama.cpp b11887 (`6050b467c`). The container
has a two-core CPU quota, 9.7 GiB RAM, and no GPU. It has no signed-in Copilot
session. Remote Strata/Maya identity tests use controlled HTTP fixtures;
SDK coverage executes pinned upstream bridge code with VS Code services stubbed.
These checks do not establish hardware inference support for every checkpoint.

## Findings and changes

- Large code prompts were substantially undercounted by the four-characters
  heuristic. A rendered 50,000-token prompt was estimated at 34,237 tokens.
  Large text requests now use llama.cpp's optional `/apply-template` and
  `/tokenize` endpoints before inference, cap the reply to remaining context,
  and reject a proven overflow with an actionable message. Token-dense text
  triggers this check using character count. Missing endpoints retain estimates
  and let the server decide whether the prompt fits. Vision requests retain
  the existing estimate path.
- Public token counting now uses the server tokenizer for large inputs, with
  bounded, short-lived caching and overlapping-request deduplication. The same
  input counted 50,000 rendered LFM tokens and 59,951 rendered Qwen tokens;
  a model-independent character estimate cannot substitute for its tokenizer.
- Explicit public API reasoning effort was overridden by VS Code's injected
  configuration defaults. Per-request `modelOptions` now wins; SDK/schema-only
  calls retain their configured selection.
- Publisher-prefixed IDs and MiniCPM5's actual `2.6B` GGUF metadata name missed
  profiles. Identity normalization fixes those cases, preserves numeric
  underscores, and excludes explicit Base/draft sidecars before alias fallback.
  FrogNano 4B 2609 gains a pinned, template-verified profile.
- During prefill the sidebar could show generation rates from an earlier call.
  It now shows “Waiting for first token…” and hides stale live generation rates.
- Externally copied models needed a manual rescan. A refresh button now rescans
  the library, restores its control after errors, and reports failures. Search
  also matches filenames when display names come from GGUF metadata.
- Advanced fields and performance cards overflowed narrow sidebars. Controls
  wrap, performance cards switch to two columns, and narrow server actions wrap.

## Live coverage and limits

Native Qwen3.5 0.8B Q4_K_M completed short English, arithmetic, Chinese, a
structured `get_weather` tool call, cancellation followed by a successful reply,
and four concurrent requests. Concurrent completion is a transport check;
this small model's answer quality is not a correctness benchmark. An oversized
59,951-token request against an 8,192-token remote slot failed before inference
in about 133 ms. Repeated public counts returned 59,939 raw text tokens.

MiniCPM5 2B and LFM2.5 230M received a 50,000-token prompt as their first
inference request through the real Language Model API. Neither produced a full
answer within this CPU test budget. MiniCPM was explicitly stopped after about
654 seconds. LFM reached 36,352 prefill tokens before the editor session ended
and the server cancelled it. The fixed UI displayed the full 50K prompt count
and waiting phase during prefill. End-to-end completion of a 50K native prompt
on representative GPU hardware remains unverified.

The initial editor session ran from approximately 04:20 to 05:14 UTC; its
interruption is excluded from active test time. A further live editor pass
starts at 08:03 UTC. The final PR records the completed cumulative duration.

A 312-case identity matrix passed across canonical IDs, filenames, publisher
prefixes, metadata, and negative cases. Ten remote model transitions exercised
the real provider against authenticated, URL-prefixed HTTP fixtures. Separate
checks passed 170 template renders across 40 checkpoints, 146 request-parser
cases across 39 Strata/Maya catalog variants, and 136 SDK bridge cases.

A scale fixture of 601 hard links to one GGUF tests filesystem/UI scale without
pretending to load 601 distinct models. The library rendered 50 rows and about
1,269 DOM nodes; a warm rescan completed in about 942 ms. Light theme and narrow
widths were checked in Chromium. Native long-prompt prefill dominated CPU use;
no inference speed or default-setting changes are inferred from shared-load
microbenchmarks. GPU throughput and display-memory reservations were not tested.
