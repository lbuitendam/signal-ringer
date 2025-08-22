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

export interface Anchor { time: number; price: number; }

export interface DrawingProps {
  color?: string; width?: number; style?: "solid" | "dash" | "dot";
  visible?: boolean; locked?: boolean; label?: string;
}
export interface DrawingMeta { createdAt: number; updatedAt: number; z: number; }

export interface Drawing {
  id: string;
  type:
    | "trendline" | "ray" | "hline" | "rect" | "path"
    | "text" | "measure" | "fib_retracement" | "fib_extension";
  anchors: Anchor[]; props: DrawingProps; meta: DrawingMeta;
}

export interface Ohlcv {
  time: number | string; open: number; high: number; low: number; close: number;
  volume?: number | null;
}

export interface OverlayPoint { time: number | string; value: number; }
export type LineDash = "solid" | "dash" | "dot";

export interface OverlaySeriesLine {
  id: string; name?: string; color?: string; width?: number; dash?: LineDash;
  data: OverlayPoint[];
}

export interface PaneLine {
  id: string; name?: string; color?: string; width?: number; dash?: LineDash;
  data: OverlayPoint[];
}
export interface PaneHist { id: string; name?: string; color?: string; data: OverlayPoint[]; }

export interface PaneSpec {
  id: string; height: number;
  yRange?: { min: number; max: number };
  lines?: PaneLine[]; hist?: PaneHist[];
  hlines?: Array<{ y: number; color?: string; dash?: LineDash }>;
}

/** Unified marker for our canvas renderer + LWC-style compat. */
export interface Marker {
  time: number | string;
  price?: number;
  side?: "above" | "below" | "inBar";
  position?: "aboveBar" | "belowBar" | "inBar";
  color?: string;
  label?: string;
}
