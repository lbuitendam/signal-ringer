import type { Anchor } from "../types";

// returns [x1,y1,x2,y2] in data space (anchors as-is)
export function asSegment(anchors: Anchor[]) { return anchors.slice(0, 2); }
