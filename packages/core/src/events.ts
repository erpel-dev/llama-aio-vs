/**
 * Minimal event plumbing shaped like `vscode.EventEmitter` so core modules can
 * be consumed unchanged by the extension, the TUI, and plain unit tests.
 */

export interface Disposable {
  dispose(): void;
}

export type Listener<T> = (value: T) => void;

export type Event<T> = (listener: Listener<T>) => Disposable;

export class Emitter<T> {
  private readonly listeners = new Set<Listener<T>>();

  readonly event: Event<T> = (listener: Listener<T>): Disposable => {
    this.listeners.add(listener);
    return {
      dispose: () => {
        this.listeners.delete(listener);
      },
    };
  };

  fire(value: T): void {
    // Copy first: a listener may unsubscribe (or subscribe) while we iterate.
    for (const listener of [...this.listeners]) {
      try {
        listener(value);
      } catch (err) {
        console.error("Llama AIO: event listener threw:", err);
      }
    }
  }

  dispose(): void {
    this.listeners.clear();
  }
}
