import type { Anchor } from "../types";

export const FIB_LEVELS_RET = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.272, 1.618];
export const FIB_LEVELS_EXT = [1, 1.272, 1.414, 1.618, 2.0];

export function fibLines(anchors: Anchor[], isExtension: boolean) {
  if (anchors.length < 2) return [];
  const [a, b] = anchors;
  const dy = b.price - a.price;
  const levels = isExtension ? FIB_LEVELS_EXT : FIB_LEVELS_RET;
  return levels.map(L => ({ level: L, price: a.price + dy * L }));
}
