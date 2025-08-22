// st_trading_draw/frontend/src/index.tsx
import React, { useEffect, useState } from "react";
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";
import { Chart } from "./Chart";
import type { Drawing, Ohlcv, OverlaySeriesLine, PaneSpec, Marker } from "./types";
import "./styles.css";

type Props = {
  args: {
    ohlcv: Ohlcv[];
    symbol: string;
    timeframe: string;
    initial_drawings: Record<string, Drawing>;
    magnet: boolean;
    toolbar_default: string;
    overlay_indicators: OverlaySeriesLine[];
    pane_indicators: PaneSpec[];
    markers: Marker[];
  };
};

class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { err?: any }> {
  constructor(props: any) { super(props); this.state = { err: undefined }; }
  componentDidCatch(err: any) { this.setState({ err }); Streamlit.setFrameHeight(); }
  render() {
    if (this.state.err) {
      return (
        <div style={{ padding: 12, fontFamily: "system-ui, sans-serif", color: "#fca5a5" }}>
          <b>Component error:</b>
          <pre style={{ whiteSpace: "pre-wrap" }}>{String(this.state.err?.message || this.state.err)}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

function useLocal<T>(key: string, init: T): [T, (v: T) => void] {
  const [v, setV] = useState<T>(() => {
    try { const s = localStorage.getItem(key); return s ? (JSON.parse(s) as T) : init; }
    catch { return init; }
  });
  const set = (x: T) => { setV(x); try { localStorage.setItem(key, JSON.stringify(x)); } catch {} };
  return [v, set];
}

const Root: React.FC<Props> = ({ args }) => {
  const {
    ohlcv = [], symbol = "", timeframe = "", initial_drawings = {},
    magnet = false, toolbar_default = "docked-right",
    overlay_indicators = [], pane_indicators = [], markers = [],
  } = args || ({} as Props["args"]);

  const [drawings, setDrawings] = useState<Record<string, Drawing>>(initial_drawings || {});
  const [activeTool, setActiveTool] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useLocal<boolean>("toolbar:collapsed", false);

  useEffect(() => { Streamlit.setComponentReady(); Streamlit.setFrameHeight(); const id = window.setInterval(() => Streamlit.setFrameHeight(), 400); return () => window.clearInterval(id); }, []);
  useEffect(() => { const onKey = (e: KeyboardEvent) => e.key === "Escape" && setActiveTool(null); window.addEventListener("keydown", onKey); return () => window.removeEventListener("keydown", onKey); }, []);
  useEffect(() => { Streamlit.setComponentValue({ drawings }); }, [drawings]);
  useEffect(() => { Streamlit.setFrameHeight(); }, [ohlcv, overlay_indicators, pane_indicators, markers, collapsed, activeTool]);

  const ToolButton = (props: { id: string; label: string }) => (
    <button className={`tool-btn ${activeTool === props.id ? "active" : ""}`} onClick={() => setActiveTool((t) => (t === props.id ? null : props.id))} title={props.label}>
      {props.label}
    </button>
  );

  return (
    <ErrorBoundary>
      <div className="root">
        <div className={`toolbar ${collapsed ? "collapsed" : ""}`}>
          <div className="toolbar-header">
            <span className="title">Tools</span>
            <button className="min-btn" onClick={() => setCollapsed(!collapsed)}>{collapsed ? "›" : "–"}</button>
          </div>
          {!collapsed && (
            <div className="toolbar-body">
              <div className="group">
                <ToolButton id="select" label="Select" />
                <ToolButton id="trendline" label="Trend" />
                <ToolButton id="ray" label="Ray" />
                <ToolButton id="rect" label="Rect" />
                <ToolButton id="path" label="Path" />
                <ToolButton id="hline" label="HLine" />
                <ToolButton id="measure" label="Measure" />
                <ToolButton id="fib_retracement" label="Fib" />
                <ToolButton id="text" label="Text" />
              </div>
              <div className="hint">ESC to cancel • Right-click to cancel a draft</div>
            </div>
          )}
        </div>

        {!collapsed && <div className="toolbar-spacer" />}

        <Chart
          data={ohlcv}
          drawings={drawings}
          setDrawings={setDrawings}
          activeTool={activeTool}
          magnet={magnet}
          toolbarKey={`${symbol}@${timeframe}`}
          overlays={overlay_indicators}
          markers={markers}
          onReady={() => {}}
        />
      </div>
    </ErrorBoundary>
  );
};

export default withStreamlitConnection(Root);
