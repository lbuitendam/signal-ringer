import type { Anchor } from "../types";
export function measure(a: Anchor, b: Anchor) {
  const dt = Math.abs(b.time - a.time);
  const dp = b.price - a.price;
  const pct = a.price !== 0 ? (dp / a.price) * 100 : 0;
  return { dt, dp, pct };
}
