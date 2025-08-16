// st_trading_draw/frontend/src/types.ts

export type ToolName =
  | "select"
  | "trendline"
  | "ray"
  | "hline"
  | "rect"
  | "path"
  | "text"
  | "measure"
  | "fib_retracement"
  | "fib_extension";

export interface Anchor {
  time: number; // unix seconds
  price: number;
}

export interface DrawingProps {
  color?: string;
  width?: number;
  style?: "solid" | "dash" | "dot";
  visible?: boolean;
  locked?: boolean;
  label?: string;
}

export interface DrawingMeta {
  createdAt: number;
  updatedAt: number;
  z: number;
}

export interface Drawing {
  id: string;
  type:
    | "trendline"
    | "ray"
    | "hline"
    | "rect"
    | "path"
    | "text"
    | "measure"
    | "fib_retracement"
    | "fib_extension";
  anchors: Anchor[];
  props: DrawingProps;
  meta: DrawingMeta;
}

export interface Ohlcv {
  time: number | string; // seconds or ISO
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number | null;
}

export interface OverlayPoint {
  time: number | string;
  value: number;
}

export interface OverlaySeriesLine {
  id: string;
  name?: string;
  color?: string;
  width?: number; // pixels
  dash?: "solid" | "dash" | "dot";
  data: OverlayPoint[];
}

export interface PaneLine {
  id: string;
  name?: string;
  color?: string;
  width?: number;
  dash?: "solid" | "dash" | "dot";
  data: OverlayPoint[];
}

export interface PaneHist {
  id: string;
  name?: string;
  color?: string;
  data: OverlayPoint[];
}

export interface PaneSpec {
  id: string;
  height: number;
  yRange?: { min: number; max: number };
  lines?: PaneLine[];
  hist?: PaneHist[];
  hlines?: Array<{ y: number; color?: string; dash?: "solid" | "dash" | "dot" }>;
}

/**
 * Unified marker type for our overlay-canvas renderer.
 * Supports BOTH our custom shape ({ price, side }) and LWC-style ({ position }).
 *
 * If price is absent, Chart will infer it from the candle:
 *  - position === 'aboveBar'  -> use candle.high
 *  - position === 'belowBar'  -> use candle.low
 *  - otherwise                -> use candle.close
 */
export interface Marker {
  time: number | string;
  // custom style
  price?: number;
  side?: "above" | "below" | "inBar";
  // LWC-style compatibility
  position?: "aboveBar" | "belowBar" | "inBar";
  // misc
  color?: string;
  label?: string; // text payload
}
