# Llama AIO (VS Code)

Local **llama.cpp** for VS Code: install newest llama.cpp binaries, manage GGUF models, tune load settings with memory estimates, run one shared `llama-server`, and use it in **GitHub Copilot Chat**.

<p align="center">
  <img src="media/context-performance.jpeg" alt="Llama AIO panel — server status, live performance, backend install, and model selection" width="49%">
  <img src="media/resource-estimate.jpeg" alt="Load settings — VRAM / RAM estimate, presets, context length, GPU offload, and CPU MoE layers" width="49%">
</p>

## llama.cpp backends

Pick a llama.cpp version in the sidebar

- **Vulkan** — default GPU path (Linux / Windows; good for AMD)
- **CUDA** — when an NVIDIA GPU is detected
- **CPU** — no GPU (`-ngl` ignored)
- **System (PATH)** — use a `llama-server` already on your `PATH` (distro / nixpkgs / self-built)

Downloaded installs live under `~/.llama-aio-vs/llama.cpp/<backend>/`.
 **Upgrade to latest release** resolves the newest tag and downloads the matching archive. New binaries are staged and swapped in at the end, so a failed install leaves the working one in place.

To pin a build or work offline: **Install release tag…** or **Install from archive…**. Browse builds at [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases).

## Models

Download models directly from Hugging Face or pick a model that was downloaded by other tools.

The picker shows each model's licence next to its name and warns before downloading anything that is not clearly permissive.

To enable vision a **mmproj** projector can be selected.

## Load settings & memory estimates

The estimate shows VRAM / RAM usage at **full context**, so you can tell whether a configuration fits before starting the server. Selecting a model auto-recommends settings: on one GPU that leaves ~2 GiB of VRAM free; on two or more GPUs it splits the model across cards first and only spills weights to system RAM if that still does not fit.

Three presets cover the common cases:

- **Coding agent** — near-lossless quality with room for tools and history
- **Max context** — the largest context that still fits your VRAM
- **Max quality** — spends VRAM on key precision instead of context

Main controls are context length, GPU offload, and CPU MoE layers. Threads, batch sizes, KV cache types, flash attention, reasoning, RoPE, and speculative decoding live under **Advanced Settings**, each group with its own reset button.

## GitHub Copilot Chat

Use local LLMs directly in GitHub Copilot Chat — the running model appears as **Llama AIO: …** in the model picker.

Sampling defaults (temperature, top-p, top-k, max tokens) are set under **Request defaults**. Well-known models such as Qwen3 ship with curated modes that override those values; the panel says so when a mode is active.

## Performance

Automated context monitoring and performance measurement of the current Chat session.

Tokens/s, context fill, and prompt reuse (how much of the prompt came from the KV cache) for the last request, also shown in the status bar. Toasts at ~80% / ~90% context fill.

## Shared server

The llama.cpp server is started from within VS Code and is shared between all running VS Code instances.

- Endpoint: `http://127.0.0.1:8742` (override with `llamaAio.host` / `llamaAio.port`)
- Lock / log: `~/.llama-aio-vs/runtime/server.lock.json`, `llama-server.log`
- Default launch: **external terminal** (sidebar **Launch** control, or `llamaAio.launchMode`); close the window to stop, or choose **Background**

Only llama-servers are ever adopted or stopped — anything else holding the port is left alone and reported instead.

## Key load settings → llama.cpp flags

| UI                              | Flag                                                                                                 |
| ------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Context Length                  | `--ctx-size`                                                                                       |
| GPU Offload                     | `-ngl`                                                                                             |
| Tensor split                    | `--tensor-split`                                                                                   |
| Split mode                      | `--split-mode`                                                                                     |
| Main GPU                        | `--main-gpu`                                                                                       |
| CPU Threads                     | `-t`                                                                                               |
| Eval / Physical batch           | `-b` / `-ub`                                                                                     |
| Max concurrent predictions      | `-np`                                                                                              |
| CPU MoE layers                  | `--n-cpu-moe`                                                                                      |
| KV cache type K / V             | `--cache-type-k` / `--cache-type-v`                                                              |
| Flash attention                 | `--flash-attn`                                                                                     |
| Unified KV cache                | `--kv-unified` / `--no-kv-unified`                                                               |
| Offload KV to GPU               | default /`--no-kv-offload`                                                                         |
| Cache reuse (KV shift)          | `--cache-reuse`                                                                                    |
| Context checkpoints             | `--ctx-checkpoints`                                                                                |
| Reasoning format / budget       | `--reasoning-format` / `--reasoning-budget`                                                      |
| Keep model in memory / Try mmap | `--load-mode mlock` / `mmap` / `none`                                                          |
| RoPE base/scale                 | `--rope-freq-base` / `--rope-freq-scale`                                                         |
| Seed                            | `--seed`                                                                                           |
| Speculative MTP                 | `--spec-type draft-mtp` (+ draft n-max/min, p-min; sidecar Gemma 4 also `-md` and `--fit off`) |
| Speculative DFlash              | `--spec-type draft-dflash` + `-md` draft GGUF (+ n-max, draft-ngl; draft KV f16)                 |

## Commands

- `Llama AIO: Open Settings`
- `Llama AIO: Install / Upgrade llama.cpp`
- `Llama AIO: Install llama.cpp by Release Tag…`
- `Llama AIO: Install llama.cpp from Archive…`
- `Llama AIO: Download Model from Hugging Face`
- `Llama AIO: Open GGUF File…`
- `Llama AIO: Choose from Downloaded Models`
- `Llama AIO: Start Server` / `Stop Server` / `Reload Server (Apply Settings)`
- `Llama AIO: Show Status`
- `Llama AIO: View Last Call` / `View Last Response`

## Settings

- `llamaAio.port` / `llamaAio.host` (defaults `8742` / `127.0.0.1`)
- `llamaAio.installDir` / `llamaAio.modelsDir` / `llamaAio.extraModelDirs`
- `llamaAio.hfToken` — gated / private HF models
- `llamaAio.autoStart`
- `llamaAio.backend` — `auto` / `vulkan` / `cuda` / `cpu` / `rocm` / `openvino` / `sycl`
- `llamaAio.launchMode` — `externalTerminal` (default) or `background`
- `llamaAio.promptReplacementsEnabled` / `llamaAio.promptReplacementsFile`License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Timo Leser
