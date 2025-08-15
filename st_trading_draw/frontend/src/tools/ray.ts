import type { Anchor } from "../types";
export function direction(a: Anchor, b: Anchor) {
  return { dx: Math.sign(b.time - a.time), dy: Math.sign(b.price - a.price) };
}
