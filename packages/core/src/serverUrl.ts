/** `scheme://host[:port]` of `url`, or undefined when it does not parse. API keys are stored per origin. */
export function originOf(url: string | undefined): string | undefined {
  if (!url) {
    return undefined;
  }
  try {
    return new URL(url).origin;
  } catch {
    return undefined;
  }
}

/**
 * Validate and canonicalize a user-entered server URL. A bare `host:port`
 * gets `http://`. Query strings and fragments are dropped; a path prefix stays.
 */
export function normalizeServerUrl(raw: string): string {
  let text = (raw || "").trim();
  if (!text) {
    throw new Error("Enter the server URL, for example http://192.168.0.5:8080.");
  }
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(text)) {
    text = `http://${text}`;
  }
  let u: URL;
  try {
    u = new URL(text);
  } catch {
    throw new Error(`“${raw.trim()}” is not a valid URL.`);
  }
  if (u.protocol !== "http:" && u.protocol !== "https:") {
    throw new Error(`Only http:// and https:// URLs are supported (got ${u.protocol}//).`);
  }
  if (!u.hostname) {
    throw new Error(`“${raw.trim()}” has no host name.`);
  }
  if (u.username || u.password) {
    throw new Error("Put the API key in the API key field, not in the URL.");
  }
  u.search = "";
  u.hash = "";
  const pathPrefix = u.pathname.replace(/\/+$/, "");
  return `${u.protocol}//${u.host}${pathPrefix}`;
}
