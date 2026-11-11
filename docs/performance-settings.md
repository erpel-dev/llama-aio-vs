# Optional llama.cpp performance settings

Available in both VS Code and the terminal UI. Changes take effect on the next local server start/reload. They do not configure a remote Strata, Project Maya, or llama.cpp server; those must be configured on the server machine. Copilot Chat and Copilot SDK/CLI requests use the same local server and benefit from its configuration.

| Saved setting (`state.loadSettings`) | UI / upstream flag | Default |
| --- | --- | --- |
| `moeCacheMiB` | GPU expert cache (MiB), `--moe-cache-mib` | `0` (off) |
| `draftSampling` | Draft sampling, `--spec-draft-sampling` | `auto` |
| `draftBatchSize` | Draft batch size, `--spec-draft-n-ubatch` | `0` (inherit main physical batch) |
| `plePath` | External PLE-only GGUF, `--shard` / `--ple` | Empty (built-in table) |

## External PLE tables

In the VS Code sidebar, select the main model, then use **PLE table → Choose…**
to attach a local PLE-only GGUF. The Command Palette also has **Llama AIO: Choose
PLE Table…**. Choose **Reload** to apply a change; **Clear** restores the built-in
table. A backbone exported without PLE requires an attachment before starting.
The terminal UI shares `state.loadSettings.plePath` in `config.json`; it does not
have a separate PLE picker yet. Switching main models clears the attachment.

Support is detected from the active executable's help (`--shard` or `--ple`),
including compatible forks/backports. Upstream first shipped it in **b12233**.
An unsupported binary refuses to start with an attachment. Remote servers manage
their own tables.

Use a table from the **same checkpoint** as the backbone. Different quantization
precisions are supported. The file must contain only `per_layer_token_embd.weight`;
Llama AIO checks its dimensions (including hashed PLE head ranges when present).
Matching dimensions cannot establish checkpoint provenance. See
[upstream conversion and loading instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md#exporting-per-layer-embeddings-separately).

Memory estimates replace the original table, count the external table once,
and recalculate MoE/FFN proportions and lazy-loading placement. A dedicated PLE
backbone shard may be absent only when the remaining declared tensor count
confirms that one tensor is missing. External PLE files are not automatically
deleted with a model, since several quants may share them.

## Compatibility

The launcher probes the exact executable's `--help` using the same environment and runtime wrapper as server start. Each flag is checked independently; probabilistic drafting also requires that the help lists the `probabilistic` choice. This supports intermediate versions and forks/backports without assuming release numbers have the same meaning everywhere.

If the executable is missing or the help probe fails/times out, these optional flags are omitted. Expert caching also requires the same binary to expose a GPU through `--list-devices`: CPU-only PATH binaries advertise the flag but would fail if it were used. Results are cached briefly by executable identity and invalidated when it is replaced. A failed help probe is retried after a short delay. The VS Code controls explain when support is unavailable; saved values survive a downgrade so they can apply again after upgrading. Copy command uses the same guard, and the server log records requested optimizations that could not be applied.

The guard applies to these new optional features. It does not make every existing Llama AIO setting compatible with arbitrarily old llama.cpp releases.

Upstream sources checked against [344be0a60074e6395da3c4a4f300f7361df5a67d](https://github.com/ggml-org/llama.cpp/blob/344be0a60074e6395da3c4a4f300f7361df5a67d/common/arg.cpp) (October 25, 2026).

## GPU expert cache

[Added October 7, 2026](https://github.com/ggml-org/llama.cpp/pull/29887), with subsequent multi-GPU and cache-policy improvements. This caches frequently used host-resident MoE experts on the GPU. It can help generation when a large MoE model cannot fit entirely in VRAM.

Set CPU MoE layers first. Only host experts in layers assigned to the GPU qualify: for partial GPU offload, the first N CPU-MoE layers must overlap the last GPU-offloaded layers. The feature is inactive for dense models, CPU-only execution, or when no such overlap exists. Upstream rejects Tensor split mode, so Llama AIO suppresses the cache in that mode as well. The saved budget remains available if the configuration becomes eligible again.

The cache duplicates weights. Original experts still occupy RAM. The memory estimate adds the full requested budget to VRAM, including warm-context totals, fit checks, and the Weights bars. For multiple GPUs, the budget is distributed only among GPUs with eligible experts. Actual allocations are layout-dependent and may be smaller; automatic splitting is approximate because upstream uses free memory at model-load time.

Leave this off until you have spare VRAM. Choose a modest explicit budget, then compare generation speed and actual VRAM use. Automatically reserving a fixed amount could force a shorter context, reduce resident model layers, or cause an allocation failure. There is no universal safe budget.

## Draft sampling

[Probabilistic MTP added October 2, 2026](https://github.com/ggml-org/llama.cpp/pull/27694). `auto` enables it for MTP, including N-gram + MTP, only when advertised by the binary. Older binaries keep their existing drafting behavior. `greedy` explicitly selects the previous strategy.

Probabilistic drafting can improve acceptance at nonzero temperature. The target verifies drafts with rejection sampling, preserving its sampling distribution. Gains vary by model, temperature, hardware, and draft length; Auto does not enable speculation for a model that does not already use MTP.

This control applies to MTP only. DFlash keeps the upstream default even if a saved MTP selection says `probabilistic`. Upstream added DFlash2 probabilistic drafting in [the October 23 extension](https://github.com/ggml-org/llama.cpp/pull/30663), but the shared flag was available before that extension. Help detection alone cannot establish DFlash2 support, so Llama AIO does not activate it for DFlash.

## Draft batch size

[Added October 14, 2026](https://github.com/ggml-org/llama.cpp/pull/30459). This sets the draft context's physical batch size separately from the main model. It applies to neural drafting (MTP or DFlash), including stacked N-gram modes, and is omitted when no neural drafter is active.

`0` inherits the main physical batch size. A smaller explicit value can reduce draft scratch-memory use, with a possible speed tradeoff. The main batch size is unchanged. The memory estimate does not assume an exact reduction in draft compute buffers from this knob; benchmark the model/backend before relying on the saving.

## Prompt-cache RAM and separate devices

Advanced settings expose three capability-checked controls:

- **Prompt cache RAM (MiB)** (`--cache-ram`): blank inherits llama.cpp's default;
  `0` disables the RAM prompt cache; `-1` allows unlimited growth. The cache grows
  on demand, separately from the live model/KV estimate; the limit is not a RAM
  reservation. This setting does not change the model's context window.
- **Projector device** (`--mmproj-device`): Auto follows the target device
  selection. Choose a detected llama.cpp GPU ID to keep the projector on a
  different GPU. The existing Projector RAM mode takes precedence.
- **Draft device** (`--spec-draft-device`): Auto follows the main model's devices;
  select one GPU or CPU for a separate MTP/DFlash draft. Embedded MTP heads
  remain attached to the target model and cannot be moved by this selector.

The active executable's help must advertise each flag. Older binaries keep
saved choices but omit unsupported overrides. CPU builds ignore saved GPU
placement, and remote servers manage their own load settings. Saved GPU IDs
that are no longer present cause a useful launch error rather than silently
placing the model on another GPU. Device selection changes the corresponding
memory bars and requires a server reload.
