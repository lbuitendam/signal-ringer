import type { Anchor } from "../types";
export function yFromAnchor(anchors: Anchor[]) { return anchors[0]?.price ?? 0; }
