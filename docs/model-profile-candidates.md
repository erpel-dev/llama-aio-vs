# Next model profiles: August 24–October 24, 2026

Snapshot checked on October 24. This shortlist prioritizes local coding and agent
use, then remote inference. It uses HF download-ranked text-generation and
image-text-to-text lists, the text-generation trending list, and author model
cards. The download figures below are HF's current rolling 30-day counts for the
linked repository, not exact two-month totals, unique users, or family totals.
Repository dates and model cards identify releases/quantizations in the requested
two-month window. Trending is a current signal, not a historical 60-day ranking.

| Candidate | HF downloads, approximately | Why add it |
| --- | ---: | --- |
| [Tiel-Coder 35B-A3B](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP) | 2.12M | Strong local coding interest; Ornith/Qwen-derived, with an embedded MTP head. Needs explicit identity matching, not sidecar exclusion. |
| [MiniCPM5 2B](https://huggingface.co/openbmb/MiniCPM5-2B) | 1.25M | Useful small-device alternative to Spark/LFM, with official GGUFs and agent/tool support. The author specifically warns that llama.cpp's default min-p .05 can cause repetition; use 0. |
| [MiMo V2.6 Distill Qwen 9B](https://huggingface.co/bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF) | 960K | Fills the gap between 2–4B and 27–35B; Q4_K_M is about 5.84GB. Verify the student's template rather than selecting a recipe from the teacher's name. |
| [Underdog Saluki 27B](https://huggingface.co/ConwayResearch/Underdog-Saluki-27B-1.0) | 535K | Sub-8GB Qwen3.8 quantization with stock llama.cpp support. Author recommends .6 temperature for thinking and 0 for direct tool calls: it needs its own sampler. |
| [Mellum2.1 12B-A2.5B Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF) | 215K | Coding-oriented reasoning MoE with official GGUFs: about 7GB MXFP4_MOE / 8.1GB Q4_K_M. Verify reasoning/tool parsing and optional MTP. |
| [Flint](https://huggingface.co/RecursiveSI/flint) / [Unsloth GGUF](https://huggingface.co/unsloth/flint-GGUF) | 79K in the GGUF repo | Highest-ranked model in the checked trending list; 39.8B total / ~2.5B active, aimed at coding agents. Native [llama.cpp support merged October 21](https://github.com/ggml-org/llama.cpp/pull/30928). Test a compatible release plus reasoning preservation and tool parsing. |
| [Sage A1.5](https://huggingface.co/InternScience/Sage-A1.5-Q4_K_M-GGUF) | 29K | New Qwen3.6-based 35B scientific/general agent model. Worth a dedicated recipe, with less adoption evidence so far than the choices above. |

My first batch would be Tiel-Coder, MiniCPM5, and Mellum2.1, followed by MiMo,
Saluki, and Flint. This balances adoption, practical model sizes, and coding use.

For remote connections, [full GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)
(2.04M) and [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
(1.57M; also investigate the newer 1017 checkpoint separately) deserve profiles.
Both are much larger than typical desktop deployments. DeepSeek's new prompt
encoding and numeric effort controls require protocol checks; do not reuse a
Qwen ladder just because the server has an OpenAI-compatible route.

[Dory 1.0 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF)
(10.5K) is interesting for small devices, but remains a watchlist item: the author
uses a custom Dory-enabled llama.cpp revision and [upstream support](https://github.com/ggml-org/llama.cpp/pull/30306)
was still open at this check. Its recurrent loop count is an engine setting,
not a reasoning-effort picker value.

The scan also caught Empero's [Qwen3.8 Distill 35B](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF)
(1.20M). Its teacher's name previously selected the dense Qwen3.8 effort ladder,
but its template is Qwen3.6-based. The current PR now gives it a separate Think
preset with the author's .6/.95/20 recipe; no unsupported effort ladder.

These recommendations are based on adoption signals and deployment suitability,
not independently reproduced author benchmark scores. Specialized memory writers,
OCR/robotics checkpoints, base models, and draft-only models were not prioritized
for a general Copilot coding provider.
