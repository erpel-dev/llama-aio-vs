/**
 * Pure text / number formatting for the sidebar. Strictly typed (tsconfig.webview.strict.json).
 */

export function escAttr(v: unknown): string {
  return String(v).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
}

export function escText(v: unknown): string {
  return String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;');
}

export function fmtEta(seconds: unknown): string {
  const s = Number(seconds) || 0;
  if (s <= 0) return '';
  if (s < 60) return s + 's left';
  if (s < 3600) return Math.round(s / 60) + ' min left';
  return (s / 3600).toFixed(1) + ' h left';
}

export function fmtCount(count: unknown): string {
  const n = Number(count) || 0;
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M downloads';
  if (n >= 1000) return Math.round(n / 1000) + 'k downloads';
  return n + ' downloads';
}

/** One unit style everywhere: GiB with one decimal, MiB below 1 GiB. */
export function fmtBytes(bytes: number | null | undefined): string {
  const GiB = 1024 ** 3, MiB = 1024 ** 2;
  if (!bytes || bytes <= 0) return '0 B';
  if (bytes >= GiB) return (bytes / GiB).toFixed(1) + ' GiB';
  if (bytes >= MiB) return (bytes / MiB).toFixed(0) + ' MiB';
  return Math.round(bytes / 1024) + ' KiB';
}

export function fmtParams(n: number): string {
  if (!(n > 0)) return '';
  if (n >= 1e9) {
    const b = n / 1e9;
    return (b >= 10 ? Math.round(b) : Math.round(b * 10) / 10) + 'B params';
  }
  if (n >= 1e6) return Math.round(n / 1e6) + 'M params';
  return String(n) + ' params';
}

export function fmtServerRate(n: number): string {
  if (!(n > 0)) return '';
  if (n >= 1000) {
    const k = n / 1000;
    return (k >= 10 ? String(Math.round(k)) : String(Math.round(k * 10) / 10)) + 'k';
  }
  return n >= 10 ? String(Math.round(n)) : String(Math.round(n * 10) / 10);
}

export function fmtTokShort(n: number): string {
  if (!(n > 0)) return '—';
  if (n >= 1048576 && n % 1048576 === 0) return (n / 1048576) + 'M';
  if (n >= 1024 && n % 1024 === 0) return (n / 1024) + 'k';
  return n >= 1000 ? Math.round(n / 1000) + 'k' : String(n);
}

export function fmtUptime(iso: string | undefined): string {
  const t = Date.parse(iso || '');
  if (!isFinite(t)) return '';
  const min = Math.max(0, Math.round((Date.now() - t) / 60000));
  if (min < 60) return 'up ' + min + ' min';
  const h = Math.floor(min / 60);
  return 'up ' + h + ' h' + (min % 60 ? ' ' + (min % 60) + ' min' : '');
}

export function fmtNum(n: unknown): string {
  return typeof n === 'number' && isFinite(n) ? Number(n).toLocaleString() : '—';
}

export function fmtTokK(n: unknown): string {
  if (typeof n !== 'number' || !isFinite(n)) return '—';
  const abs = Math.abs(n);
  if (abs >= 1000) {
    const digits = abs >= 10000 ? 0 : 1;
    const s = (n / 1000).toFixed(digits);
    return (s.endsWith('.0') ? s.slice(0, -2) : s) + 'k';
  }
  return Math.round(n).toLocaleString();
}

export function fmtRate(n: unknown): string | undefined {
  if (typeof n !== 'number' || !isFinite(n) || n <= 0) return undefined;
  return n >= 100 ? n.toFixed(0) : n.toFixed(1);
}

export function niceCeil(n: number): number {
  if (!(n > 0)) return 20;
  if (n <= 20) return 20;
  if (n <= 100) return Math.ceil(n / 10) * 10;
  return Math.ceil(n / 100) * 100;
}

export function maxPositive(values: ArrayLike<unknown>): number {
  let m = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (typeof v === 'number' && isFinite(v) && v > m) m = v;
  }
  return m;
}

export function pickNum(...values: unknown[]): number | undefined {
  return values.find((n): n is number => typeof n === 'number' && isFinite(n));
}
