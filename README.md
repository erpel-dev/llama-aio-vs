# Llama AIO (VS Code)

All-in-one local **llama.cpp** extension for VS Code:

- Install / upgrade `llama.cpp` (Vulkan, CUDA, or CPU — each kept in its own folder)
- Browse & download GGUF models from Hugging Face
- Open local GGUFs or pick from common download caches (LM Studio, Unsloth, HF cache, …)
- LM Studio–style load settings with memory estimates and auto-recommended defaults on model select
- Start a **single shared** detached `llama-server` (survives window/folder close)
- Register the model in **Copilot Chat** via `LanguageModelChatProvider`

## Requirements

- VS Code **1.109+** (for language model chat providers)
- Network access to download llama.cpp releases and HF models
- **Windows:** PowerShell (for zip extract / external terminal). CUDA installs also fetch the matching `cudart` DLL package.

## Quick start

```bash
cd llama-aio-vs
make          # compiles TypeScript and builds the .vsix
# or: make install   # also installs into VS Code / Cursor
```

Then install the generated `llama-aio-vs-*.vsix` via **Extensions: Install from VSIX…**, or use `make install`. Reload the window after installing.

1. Open the **Llama AIO** activity bar panel
2. **Install / switch backend** (once per backend you use)
3. Get a model using one of:
   - **Download from Hugging Face…**
   - **Open GGUF file…**
   - **Choose from downloaded…** (scans Llama AIO, LM Studio, Unsloth Studio, HF cache, GPT4All, Jan, …)
4. Tune load settings → **Start server** / **Reload Server (Apply Settings)**
5. In Copilot Chat, select **Llama AIO: …** in the model picker (a toast offers **Open Chat** / **Open model picker** after start)

## llama.cpp version & backends

The sidebar **llama.cpp** card shows the installed build (`llama-server --version`, release tag, asset name).

Sidebar backends:

- **Vulkan** — default GPU path on Linux / Windows (good for AMD)
- **CUDA** — only offered when an NVIDIA GPU is detected
- **CPU** — no GPU (`-ngl` ignored)

Each backend installs under `~/.llama-aio-vs/llama.cpp/<backend>/` (e.g. `vulkan`, `cuda`, `cpu`). Switching reuses a cached install when present; use **Upgrade to latest release** to re-download. Additional families (`rocm`, `openvino`, `sycl`, `auto`) are available via the `llamaAio.backend` setting.

## Load settings & memory estimates

Primary controls (always visible):

- **Memory estimate** — weights / KV / overhead on VRAM vs system RAM; red warning when estimated GPU use leaves too little headroom (~92%)
- **Context Length**, **GPU Offload**, **CPU MoE layers**

Selecting a new model auto-recommends context (up to 65 536 when the model allows), GPU layers, and MoE CPU offload aimed at ~2 GiB free VRAM. Threads, batch sizes, KV/mmap/RoPE, and speculative decoding sit under collapsed **Advanced Settings**.

KV estimates for hybrid models (e.g. Qwen3.5) count only full-attention layers when metadata provides that.

## Performance & context meter

After each Copilot Chat reply (and roughly at request start):

- **Status bar:** tok/s and context fill (`72% ctx`); warning/error colors at 80% / 90%
- **Sidebar Performance:** bar + `Context: 48k / 65k (74%)`
- Toast when crossing ~80% (info) and ~90% (warning)

Context % is `prompt_tokens / slot n_ctx` for the last request (conversation + tools). It is not a live in-chat Copilot widget.

## Shared server design

- Default endpoint: `http://127.0.0.1:8742`
- Lock file: `~/.llama-aio-vs/runtime/server.lock.json`
- Log: `~/.llama-aio-vs/runtime/llama-server.log`
- Default launch: **external terminal** (`llamaAio.launchMode`: `externalTerminal`)
  - Live logs in a system terminal window
  - Closing that window stops the server
  - Set `background` to hide the process instead
- Not killed on VS Code deactivate (unless you close the terminal / use Stop)
- Starting again reuses an existing healthy server on the configured port when settings match

## Key load settings → llama.cpp flags

| UI | Flag |
|---|---|
| Context Length | `--ctx-size` |
| GPU Offload | `-ngl` |
| CPU Threads | `-t` |
| Eval / Physical batch | `-b` / `-ub` |
| Max concurrent predictions | `-np` |
| CPU MoE layers | `--n-cpu-moe` |
| Offload KV to GPU | default / `--no-kv-offload` |
| Keep model in memory / Try mmap | `--load-mode mlock` / `mmap` / `none` |
| RoPE base/scale | `--rope-freq-base` / `--rope-freq-scale` |
| Seed | `--seed` |
| Context checkpoints | `--cache-reuse` |
| Speculative MTP | `--spec-type draft-mtp` (+ `--spec-draft-n-max/min`, `--spec-draft-p-min`) |

## Commands

- `Llama AIO: Open Settings`
- `Llama AIO: Install / Upgrade llama.cpp`
- `Llama AIO: Download Model from Hugging Face`
- `Llama AIO: Open GGUF File…`
- `Llama AIO: Choose from Downloaded Models`
- `Llama AIO: Start Server`
- `Llama AIO: Stop Server`
- `Llama AIO: Reload Server (Apply Settings)`
- `Llama AIO: Show Status`

## Settings

- `llamaAio.port` (default `8742`)
- `llamaAio.host` (default `127.0.0.1`)
- `llamaAio.installDir` / `llamaAio.modelsDir` overrides
- `llamaAio.extraModelDirs` — extra folders for **Choose from downloaded…**
- `llamaAio.hfToken` — gated / private HF models
- `llamaAio.autoStart`
- `llamaAio.backend` — `auto` / `vulkan` / `cuda` / `cpu` / `rocm` / `openvino` / `sycl`
- `llamaAio.launchMode` — `externalTerminal` (default) or `background`

## Notes

- **Keep Model in Memory** pins weights via `--load-mode mlock`; it does not control process lifetime. Lifetime comes from the shared server / terminal window.
- MTP needs a GGUF with next-n / MTP layers (e.g. Ornith MTP). Plain models often lack that metadata.
- Copilot **Agent** mode sends very large prompts; prefer GPU backends or Ask mode with a smaller context when testing on CPU.
- Focus is chat provider + shared server + settings UI (not FIM/RAG/agent UI).

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Timo Leser
