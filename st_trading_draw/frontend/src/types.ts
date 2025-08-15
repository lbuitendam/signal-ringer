export type DrawingType =
  | "fib_retracement" | "fib_extension"
  | "trendline" | "ray" | "hline"
  | "rect" | "path" | "text" | "measure";

export interface Anchor { time: number; price: number } // unix seconds
export type LineStyle = "solid" | "dash" | "dot";

export interface Drawing {
  id: string;
  type: DrawingType;
  anchors: Anchor[];
  props: {
    color: string; width: number; style: LineStyle;
    extendLeft?: boolean; extendRight?: boolean;
    visible: boolean; locked: boolean; label?: string;
  };
  meta: { createdAt: number; updatedAt: number; z?: number };
}

export interface Ohlcv { time: number; open: number; high: number; low: number; close: number; volume?: number }

export interface PayloadIn {
  ohlcv: Ohlcv[];
  symbol: string;
  timeframe: string;
  initial_drawings: Record<string, Drawing>;
  magnet: boolean;
  toolbar_default: "floating" | "docked-left" | "docked-right";
}

export interface PayloadOut {
  drawings: Record<string, Drawing>;
  toolbar: { mode: "floating" | "docked-left" | "docked-right"; x: number; y: number };
}
