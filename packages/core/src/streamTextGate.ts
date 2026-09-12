/**
 * Live-streaming filter for assistant text while tools are enabled.
 *
 * llama.cpp chat templates without native tool-call support (Qwen/Hermes
 * style) put `<tool_call>` XML into `delta.content`. Buffering the whole
 * response to catch that made Copilot Agent mode look frozen for the entire
 * generation. This gate forwards text as it arrives and only holds back the
 * few trailing characters that could still turn into a `<tool_call` or
 * `<think>` opener. Once a tool-call opener is seen the gate halts; the caller
 * reconciles the tail against the fully parsed response in {@link finish}.
 *
 * Thinking blocks (`<think>…</think>`) are removed from the live stream to
 * match `stripThinkTags` in the final pass.
 */

const TOOL_OPEN_RE = /<tool_call\b/i;
const THINK_OPEN_RE = /<think(?:ing)?\b[^>]*>/i;
const THINK_CLOSE_RE = /<\/think(?:ing)?\b[^>]*>/i;

/** Longest text after a `<` that we are willing to withhold as a possible tag. */
const MAX_HELD_TAG = 64;

const TAG_PREFIXES = ["<tool_call", "<thinking", "</thinking"];

/** Length of a trailing fragment that could still grow into a tag we act on. */
export function partialTagTailLength(text: string): number {
  const lt = text.lastIndexOf("<");
  if (lt < 0) {
    return 0;
  }
  const tail = text.slice(lt);
  if (tail.length > MAX_HELD_TAG || tail.includes(">")) {
    return 0;
  }
  const lower = tail.toLowerCase();
  if (TAG_PREFIXES.some((p) => p.startsWith(lower))) {
    return tail.length;
  }
  // `<think foo=` — an opened think tag whose `>` has not arrived yet.
  if (/^<\/?think(?:ing)?\b[^>]*$/i.test(tail)) {
    return tail.length;
  }
  return 0;
}

export class LiveTextGate {
  private pending = "";
  private inThink = false;
  private halted = false;
  private emitted = "";

  /** True once a `<tool_call` opener was seen; nothing more is streamed live. */
  get isHalted(): boolean {
    return this.halted;
  }

  /** Everything handed out so far (after think-block removal). */
  get emittedText(): string {
    return this.emitted;
  }

  /** Feed a content delta; returns the text that is safe to show right now. */
  push(chunk: string): string {
    if (this.halted || !chunk) {
      return "";
    }
    this.pending += chunk;
    let out = "";
    for (;;) {
      if (this.inThink) {
        const close = THINK_CLOSE_RE.exec(this.pending);
        if (!close) {
          const hold = partialTagTailLength(this.pending);
          this.pending = hold ? this.pending.slice(-hold) : "";
          break;
        }
        this.pending = this.pending.slice(close.index + close[0].length);
        this.inThink = false;
        continue;
      }

      const tool = TOOL_OPEN_RE.exec(this.pending);
      const open = THINK_OPEN_RE.exec(this.pending);
      const close = THINK_CLOSE_RE.exec(this.pending);
      const candidates: Array<{ at: number; len: number; kind: "tool" | "open" | "close" }> = [];
      if (tool) {
        candidates.push({ at: tool.index, len: tool[0].length, kind: "tool" });
      }
      if (open) {
        candidates.push({ at: open.index, len: open[0].length, kind: "open" });
      }
      if (close) {
        candidates.push({ at: close.index, len: close[0].length, kind: "close" });
      }
      if (!candidates.length) {
        const hold = partialTagTailLength(this.pending);
        const safe = this.pending.length - hold;
        out += this.pending.slice(0, safe);
        this.pending = this.pending.slice(safe);
        break;
      }
      candidates.sort((a, b) => a.at - b.at);
      const first = candidates[0]!;
      out += this.pending.slice(0, first.at);
      if (first.kind === "tool") {
        this.pending = "";
        this.halted = true;
        break;
      }
      this.pending = this.pending.slice(first.at + first.len);
      if (first.kind === "open") {
        this.inThink = true;
      }
    }
    return this.record(out);
  }

  /**
   * Called once the stream ended. `finalVisible` is the fully cleaned text
   * (think tags and tool-call XML stripped). Returns whatever still needs to
   * be shown so live output + this remainder equals `finalVisible`.
   */
  finish(finalVisible: string): string {
    if (!this.halted) {
      const rest = this.inThink ? "" : this.pending;
      this.pending = "";
      if (rest) {
        return this.record(rest);
      }
      if (!this.emitted && finalVisible) {
        // Nothing streamed (e.g. reasoning-only reply) — show the final text.
        return this.record(finalVisible);
      }
      return "";
    }
    const done = this.emitted.trim();
    if (!done) {
      return this.record(finalVisible);
    }
    if (finalVisible.startsWith(this.emitted)) {
      return this.record(finalVisible.slice(this.emitted.length));
    }
    if (finalVisible.startsWith(done)) {
      return this.record(finalVisible.slice(done.length));
    }
    // Live text and the parsed result disagree (should not happen); never
    // duplicate what the user already saw.
    return "";
  }

  private record(text: string): string {
    if (!text) {
      return "";
    }
    const out = this.emitted ? text : text.replace(/^\s+/, "");
    this.emitted += out;
    return out;
  }
}
