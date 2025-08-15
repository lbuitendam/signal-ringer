import type { Anchor } from "../types";
export function position(anchors: Anchor[]) { return anchors[0] ?? {time:0,price:0}; }
