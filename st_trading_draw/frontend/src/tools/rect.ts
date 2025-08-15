import type { Anchor } from "../types";
export function rectBounds(anchors: Anchor[]) {
  const [a,b] = anchors; if(!a||!b) return null;
  const t1 = Math.min(a.time,b.time), t2=Math.max(a.time,b.time);
  const y1 = Math.min(a.price,b.price), y2=Math.max(a.price,b.price);
  return { t1,t2,y1,y2 };
}
