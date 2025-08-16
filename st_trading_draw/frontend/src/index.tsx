import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom";
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";
import { Chart } from "./Chart";
import { Toolbar, type ToolName } from "./Toolbar";
import IndicatorsPane from "./IndicatorsPane";
import type {
  Drawing,
  Ohlcv,
  OverlaySeriesLine,
  PaneSpec,
  Marker,
} from "./types";
import type { IChartApi } from "lightweight-charts";

type Args = {
  // base
  ohlcv?: Ohlcv[];
  data?: Ohlcv[]; // alias
  symbol?: string;
  timeframe?: string;

  // drawings
  initial_drawings?: Record<string, Drawing>;
  magnet?: boolean;
  toolbar_default?: "docked-right" | "docked-left" | "floating";
  toolbarKey?: string;
  activeTool?: ToolName | null;

  // indicators & panes
  overlay_indicators?: OverlaySeriesLine[];
  pane_indicators?: PaneSpec[];

  // patterns
  markers?: Marker[];
};

const Component = (props: any) => {
  const args = (props.args || {}) as Args;

  const data = useMemo<Ohlcv[]>(
    () =>
      Array.isArray(args.data)
        ? args.data
        : Array.isArray(args.ohlcv)
        ? args.ohlcv
        : [],
    [args.data, args.ohlcv]
  );

  const overlays = useMemo<OverlaySeriesLine[]>(
    () =>
      Array.isArray(args.overlay_indicators) ? args.overlay_indicators : [],
    [args.overlay_indicators]
  );

  const panes = useMemo<PaneSpec[]>(
    () => (Array.isArray(args.pane_indicators) ? args.pane_indicators : []),
    [args.pane_indicators]
  );

  const markers = useMemo<Marker[]>(
    () => (Array.isArray(args.markers) ? args.markers : []),
    [args.markers]
  );

  // drawings
  const [drawings, setDrawings] = useState<Record<string, Drawing>>(
    args.initial_drawings || {}
  );
  useEffect(() => {
    Streamlit.setComponentValue({ drawings });
  }, [drawings]);

  // tool & dock
  const [activeTool, setActiveTool] = useState<ToolName | null>(
    typeof args.activeTool === "undefined" ? null : args.activeTool
  );
  useEffect(() => {
    if (typeof args.activeTool !== "undefined") setActiveTool(args.activeTool);
  }, [args.activeTool]);

  const defaultDock =
    args.toolbar_default === "docked-left"
      ? "left"
      : args.toolbar_default === "floating"
      ? "floating"
      : "right";
  const [dock, setDock] = useState<"left" | "right" | "floating">(defaultDock);

  const magnet = !!args.magnet;
  const toolbarKey =
    args.toolbarKey || `${args.symbol || "SYMBOL"}@${args.timeframe || "TF"}`;

  // main chart ref for pane sync
  const mainChartRef = useRef<IChartApi | null>(null);

  // dynamic height: 560 for main + sum of pane heights
  const totalHeight =
    560 + (panes?.reduce((acc, p) => acc + (p.height || 0), 0) || 0);

  useEffect(() => {
    Streamlit.setFrameHeight(totalHeight);
  }, [totalHeight, panes?.length]);

  return (
    <div style={{ position: "relative", width: "100%", height: totalHeight }}>
      {/* Main chart */}
      <div style={{ position: "relative", width: "100%", height: 560 }}>
        <Chart
          data={data}
          drawings={drawings}
          setDrawings={setDrawings}
          activeTool={activeTool}
          magnet={magnet}
          toolbarKey={toolbarKey}
          overlays={overlays}
          markers={markers}
          onReady={(api) => {
            mainChartRef.current = api;
          }}
        />
        <Toolbar
          activeTool={activeTool}
          onSelect={setActiveTool}
          dock={dock}
          setDock={setDock}
          canClear={Object.keys(drawings).length > 0}
          onClear={() => setDrawings({})}
          magnet={magnet}
        />
      </div>

      {/* Indicator panes (RSI / MACD / etc) */}
      {panes.map((p) => (
        <div key={p.id} style={{ width: "100%", height: p.height }}>
          <IndicatorsPane pane={p} syncWith={mainChartRef.current || undefined} />
        </div>
      ))}
    </div>
  );
};

const Wrapped = withStreamlitConnection(Component);
ReactDOM.render(<Wrapped />, document.getElementById("root"));
Streamlit.setComponentReady();
