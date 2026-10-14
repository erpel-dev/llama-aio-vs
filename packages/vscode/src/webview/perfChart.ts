/**
 * Per-tile sparklines for the performance tiles (header and Performance card).
 * One series, one colour, its own scale: the trend is the point, the number
 * sits in the tile above it. Replaces the dual-axis chart, which was hard to
 * read at sidebar width. Strictly typed.
 */
import { maxPositive } from "./format";

type Series = ArrayLike<number | null | undefined>;

/**
 * Inline SVG polyline over the positive values of `values` (oldest first).
 * Empty string when fewer than two usable points exist.
 */
export function sparklineSvg(values: Series, color: string, width = 80, height = 14): string {
  const pts: Array<{ i: number; v: number }> = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (typeof v === 'number' && isFinite(v) && v > 0) pts.push({ i, v });
  }
  if (pts.length < 2) return '';
  const max = maxPositive(values);
  let min = max;
  for (const p of pts) min = Math.min(min, p.v);
  // Keep flat series visible instead of collapsing onto one edge.
  const span = max - min || max || 1;
  const n = Math.max(1, values.length - 1);
  const pad = 1.5;
  const coords = pts
    .map((p) => {
      const x = (p.i / n) * (width - 2 * pad) + pad;
      const y = height - pad - ((p.v - min) / span) * (height - 2 * pad);
      return x.toFixed(1) + ',' + y.toFixed(1);
    })
    .join(' ');
  const last = coords.split(' ').pop()!.split(',');
  return (
    '<svg xmlns="http://www.w3.org/2000/svg" class="spark-svg" width="' + width + '" height="' + height +
    '" viewBox="0 0 ' + width + ' ' + height + '" aria-hidden="true">' +
    '<polyline points="' + coords + '" fill="none" stroke="' + color + '" stroke-width="1.4" stroke-linejoin="round"/>' +
    '<circle cx="' + last[0] + '" cy="' + last[1] + '" r="1.8" fill="' + color + '"/>' +
    '<' + '/svg>'
  );
}
