import React, { useEffect, useRef, useState } from "react";
import {
  createChart,
  type IChartApi,
  type UTCTimestamp,
  type CandlestickData,
  type ISeriesApi,
} from "lightweight-charts";
import type { Anchor, Drawing, Ohlcv, OverlaySeriesLine, Marker } from "./types";
import { fibLines } from "./tools/fibonacci";
import { rectBounds } from "./tools/rect";
import { measure } from "./tools/measure";

function unix(t: number): UTCTimestamp {
  return Math.floor(t) as UTCTimestamp;
}

export interface ChartProps {
  data: Ohlcv[];
  drawings: Record<string, Drawing>;
  setDrawings: (d: Record<string, Drawing>) => void;
  activeTool: string | null;
  magnet: boolean;
  toolbarKey: string;
  overlays?: OverlaySeriesLine[];
  markers?: Marker[];
  onReady?: (api: IChartApi) => void;
}

export function Chart({
  data,
  drawings,
  setDrawings,
  activeTool,
  magnet,
  toolbarKey,
  overlays = [],
  markers = [],
  onReady,
}: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const normDataRef = useRef<CandlestickData[]>([]);
  const overlaySeriesMap = useRef<Map<string, ISeriesApi<"Line">>>(new Map());

  const [draft, setDraft] = useState<Anchor[]>([]);
  const [hover, setHover] = useState<Anchor | null>(null);

  // Control: draw labels on pattern markers on the canvas?
  const DRAW_MARKER_LABELS = false;

  // ---- helpers ----
  function toLWTime(t: number | string): number {
    if (typeof t === "number") return t > 1e12 ? Math.floor(t / 1000) : Math.floor(t);
    const ms = Date.parse(t);
    return Math.floor(ms / 1000);
  }

  function normalizeData(rows: any[]): CandlestickData[] {
    if (!Array.isArray(rows)) return [];
    const out = rows
      .filter(
        (r) =>
          r &&
          r.time != null &&
          Number.isFinite(r.open) &&
          Number.isFinite(r.high) &&
          Number.isFinite(r.low) &&
          Number.isFinite(r.close)
      )
      .map((r) => ({
        time: toLWTime(r.time) as UTCTimestamp,
        open: +r.open,
        high: +r.high,
        low: +r.low,
        close: +r.close,
      }));
    out.sort((a, b) => (a.time as number) - (b.time as number));
    return out;
  }

  function findNearestIndexByTime(t: number): number {
    const arr = normDataRef.current;
    if (!arr.length) return 0;
    let lo = 0,
      hi = arr.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if ((arr[mid].time as number) < t) lo = mid + 1;
      else hi = mid;
    }
    const i = lo;
    if (i === 0) return 0;
    const t0 = arr[i - 1].time as number;
    const t1 = arr[i].time as number;
    return Math.abs(t - t0) <= Math.abs(t1 - t) ? i - 1 : i;
  }

  function snapAnchor(time: number, price: number): Anchor {
    const arr = normDataRef.current;
    if (!arr.length) return { time, price };
    const i = findNearestIndexByTime(time);
    const c = arr[i];
    const candidates = [c.open, c.high, c.low, c.close];
    let best = candidates[0],
      bd = Math.abs(price - candidates[0]);
    for (let k = 1; k < candidates.length; k++) {
      const d = Math.abs(price - candidates[k]);
      if (d < bd) {
        bd = d;
        best = candidates[k];
      }
    }
    return { time: c.time as number, price: best };
  }

  // === Stable coordinate: prefer logical index -> x, price -> y ===
  function xyAt(time: number, price: number) {
    const chart = chartRef.current;
    const s = seriesRef.current;
    const arr = normDataRef.current;
    if (!chart || !s || !arr.length) return null;

    const i = findNearestIndexByTime(time);
    const ts: any = chart.timeScale();

    const x =
      typeof ts.logicalToCoordinate === "function"
        ? (ts.logicalToCoordinate(i) as number | null)
        : (ts.timeToCoordinate(arr[i].time as number) as number | null);

    if (x == null) return null;

    const y = s.priceToCoordinate(price) ?? null;
    if (y == null) return null;

    return { x, y, i };
  }

  function toXY(a: Anchor) {
    return xyAt(a.time, a.price);
  }

  // Batch paints
  let raf: number | null = null;
  function scheduleDraw() {
    if (raf != null) return;
    raf = requestAnimationFrame(() => {
      raf = null;
      drawAll();
    });
  }

  // ---- mount main chart ----
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    if (!el.style.minHeight) el.style.minHeight = "420px";
    el.style.position = "relative";

    const chart = createChart(el, {
      width: el.clientWidth,
      height: el.clientHeight || 520,
      layout: { background: { color: "#0e1117" }, textColor: "#e5e7eb" },
      timeScale: { rightOffset: 2, borderColor: "#333" },
      grid: { vertLines: { color: "#1f2937" }, horzLines: { color: "#1f2937" } },
      handleScroll: { mouseWheel: true, pressedMouseMove: true },
      handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: { time: true, price: true } },
    });

    const s = chart.addCandlestickSeries({
      upColor: "#26a69a",
      downColor: "#ef5350",
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
      borderVisible: false,
    });

    chartRef.current = chart;
    seriesRef.current = s;
    onReady?.(chart);

    // overlay canvas sizing
    const overlay = overlayRef.current;
    const sizeOverlay = () => {
      if (!overlay || !el) return;
      const rect = el.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      overlay.width = Math.floor(rect.width * dpr);
      overlay.height = Math.floor(rect.height * dpr);
      overlay.style.width = rect.width + "px";
      overlay.style.height = rect.height + "px";
      const ctx = overlay.getContext("2d");
      if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    sizeOverlay();

    const ro = new ResizeObserver(() => {
      chart.applyOptions({ width: el.clientWidth, height: el.clientHeight || 520 });
      sizeOverlay();
      scheduleDraw();
    });
    ro.observe(el);

    const onRangeChange = () => {
      sizeOverlay();
      scheduleDraw();
    };
    chart.timeScale().subscribeVisibleTimeRangeChange(onRangeChange);
    const tsAny = chart.timeScale() as any;
    const hasLogical =
      typeof tsAny.subscribeVisibleLogicalRangeChange === "function" &&
      typeof tsAny.unsubscribeVisibleLogicalRangeChange === "function";
    if (hasLogical) tsAny.subscribeVisibleLogicalRangeChange(onRangeChange);

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setDraft([]);
        setHover(null);
        scheduleDraw();
      }
    };
    window.addEventListener("keydown", onKey);

    return () => {
      window.removeEventListener("keydown", onKey);
      ro.disconnect();
      chart.timeScale().unsubscribeVisibleTimeRangeChange(onRangeChange);
      if (hasLogical) tsAny.unsubscribeVisibleLogicalRangeChange(onRangeChange);
      for (const s of overlaySeriesMap.current.values()) chart.removeSeries(s);
      overlaySeriesMap.current.clear();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      normDataRef.current = [];
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- candles update ----
  useEffect(() => {
    const s = seriesRef.current;
    if (!s) return;
    const norm = normalizeData(data);
    if (!norm.length) return;
    normDataRef.current = norm;
    s.setData(norm);
    scheduleDraw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  // ---- overlays update ----
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const nextIds = new Set(overlays.map((o) => o.id));
    for (const [id, s] of overlaySeriesMap.current) {
      if (!nextIds.has(id)) {
        chart.removeSeries(s);
        overlaySeriesMap.current.delete(id);
      }
    }
    for (const o of overlays) {
      const lw = Math.max(1, Math.round((o.width ?? 2) as number)) as any;
      const ls = (o.dash === "dash" ? 1 : o.dash === "dot" ? 2 : 0) as any;
      let line = overlaySeriesMap.current.get(o.id);
      if (!line) {
        line = chart.addLineSeries({
          color: o.color || "#33cccc",
          lineWidth: lw as any,
          lineStyle: ls,
          priceLineVisible: false,
        });
        overlaySeriesMap.current.set(o.id, line);
      } else {
        line.applyOptions({ color: o.color || "#33cccc", lineWidth: lw as any, lineStyle: ls } as any);
      }
      const lineData = (o.data || [])
        .filter((r) => r && r.time != null && Number.isFinite(r.value))
        .map((r) => ({ time: toLWTime(r.time) as UTCTimestamp, value: +r.value }))
        .sort((a, b) => (a.time as number) - (b.time as number));
      if (lineData.length) line.setData(lineData);
    }
    scheduleDraw();
  }, [overlays]);

  // ---- markers changed ----
  useEffect(() => {
    scheduleDraw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [markers]);

  // ---- drawing mode toggles ----
  useEffect(() => {
    const overlay = overlayRef.current;
    const chart = chartRef.current as any;
    const drawingActive = !!(activeTool && activeTool !== "select");
    if (overlay) {
      overlay.style.pointerEvents = drawingActive ? "auto" : "none";
      overlay.style.cursor = drawingActive ? "crosshair" : "default";
    }
    if (chart?.applyOptions) {
      chart.applyOptions({
        handleScroll: drawingActive
          ? { mouseWheel: false, pressedMouseMove: false, horzTouchDrag: false, vertTouchDrag: false }
          : { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
        handleScale: drawingActive
          ? { mouseWheel: false, pinch: false, axisPressedMouseMove: { time: false, price: false } }
          : { mouseWheel: true, pinch: true, axisPressedMouseMove: { time: true, price: true } },
      });
    }
  }, [activeTool]);

  function addDrawing(d: Drawing) {
    const copy = { ...drawings, [d.id]: d };
    setDrawings(copy);
    try {
      localStorage.setItem(`drawings:${toolbarKey}`, JSON.stringify(copy));
    } catch {}
  }

  function getPriceForMarker(m: Marker): number | null {
    const t = typeof m.time === "string" ? toLWTime(m.time) : (m.time as number);
    const arr = normDataRef.current;
    if (!arr.length) return null;
    const i = findNearestIndexByTime(t);
    const bar = arr[i];
    if (!bar) return null;

    if (typeof m.price === "number" && Number.isFinite(m.price)) return m.price;

    const pos =
      m.side ??
      (m.position === "aboveBar" ? "above" : m.position === "belowBar" ? "below" : "inBar");
    if (pos === "above") return bar.high;
    if (pos === "below") return bar.low;
    return bar.close;
  }

  // ---- overlay canvas paint: drawings + previews + markers ----
  function drawAll() {
    const canvas = overlayRef.current,
      chart = chartRef.current,
      s = seriesRef.current;
    if (!canvas || !chart || !s) return;
    const ctx = canvas.getContext("2d")!;
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);

    // drawings
    const items = Object.values(drawings).filter((d) => d.props.visible !== false);
    for (const d of items) {
      ctx.lineWidth = d.props.width ?? 1.5;
      ctx.strokeStyle = d.props.color ?? "#eab308";
      ctx.setLineDash(d.props.style === "dash" ? [6, 4] : d.props.style === "dot" ? [2, 4] : []);

      if (d.type === "trendline" || d.type === "ray") {
        const [a, b] = d.anchors;
        if (!a || !b) continue;
        const A = toXY(a),
          B = toXY(b);
        if (!A || !B) continue;
        ctx.beginPath();
        ctx.moveTo(A.x, A.y);
        if (d.type === "ray") {
          const dx = B.x - A.x,
            dy = B.y - A.y,
            k = (rect.width - A.x) / (dx || 1e-6);
          ctx.lineTo(A.x + dx * k, A.y + dy * k);
        } else {
          ctx.lineTo(B.x, B.y);
        }
        ctx.stroke();
      } else if (d.type === "hline") {
        const y = toXY(d.anchors[0])?.y;
        if (y == null) continue;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(rect.width, y);
        ctx.stroke();
      } else if (d.type === "rect") {
        const rb = rectBounds(d.anchors);
        if (!rb) continue;
        const A = toXY({ time: rb.t1, price: rb.y1 });
        const B = toXY({ time: rb.t2, price: rb.y2 });
        if (!A || !B) continue;
        const x = A.x,
          y = B.y,
          w = B.x - A.x,
          h = A.y - B.y;
        ctx.save();
        ctx.globalAlpha = 0.12;
        ctx.fillStyle = (d.props.color as string) || "#eab308";
        ctx.fillRect(x, y, w, h);
        ctx.restore();
        ctx.strokeRect(x, y, w, h);
      } else if (d.type === "path") {
        const pts = d.anchors.map((a) => toXY(a)).filter(Boolean) as Array<{ x: number; y: number }>;
        if (pts.length < 2) continue;
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        ctx.stroke();
      } else if (d.type === "text") {
        const p = toXY(d.anchors[0]);
        if (!p) continue;
        ctx.fillStyle = (d.props.color as string) || "#e5e7eb";
        ctx.font = "12px Inter, system-ui, sans-serif";
        ctx.fillText(d.props.label || "Text", p.x + 4, p.y - 4);
      } else if (d.type === "measure") {
        const [a, b] = d.anchors;
        if (!a || !b) continue;
        const A = toXY(a),
          B = toXY(b);
        if (!A || !B) continue;
        ctx.beginPath();
        ctx.moveTo(A.x, A.y);
        ctx.lineTo(B.x, B.y);
        ctx.stroke();
        const m = measure(a, b);
        const label = `Δp: ${m.dp.toFixed(4)} (${m.pct.toFixed(2)}%)  Δt: ${m.dt}s`;
        ctx.fillStyle = "#e5e7eb";
        ctx.fillText(label, (A.x + B.x) / 2 + 6, (A.y + B.y) / 2 - 6);
      } else if (d.type === "fib_retracement" || d.type === "fib_extension") {
        const lines = fibLines(d.anchors, d.type === "fib_extension");
        if (lines.length === 0) continue;
        const [a, b] = d.anchors;
        const t1 = Math.min(a.time, b.time);
        const t2 = Math.max(a.time, b.time);
        for (const lv of lines) {
          const Lp = toXY({ time: t1, price: lv.price });
          const Rp = toXY({ time: t2, price: lv.price });
          if (!Lp || !Rp) continue;
          ctx.beginPath();
          ctx.moveTo(Lp.x, Lp.y);
          ctx.lineTo(Rp.x, Rp.y);
          ctx.stroke();
          ctx.fillStyle = (d.props.color as string) || "#e5e7eb";
          ctx.fillText(`${(lv.level * 100).toFixed(1)}%  ${lv.price.toFixed(4)}`, Rp.x + 6, Rp.y - 2);
        }
      }

      // small anchors
      for (const a of d.anchors) {
        const p = toXY(a);
        if (!p) continue;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = "#e5e7eb";
        ctx.fill();
      }
    }

    // live preview while drafting
    if (activeTool && activeTool !== "select" && draft.length) {
      const color = "#a3e635";
      const first = toXY(draft[0]);
      if (first) {
        ctx.beginPath();
        ctx.arc(first.x, first.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
      if (hover) {
        const A = toXY(draft[0]);
        const B = toXY(hover);
        if (A && B) {
          ctx.beginPath();
          ctx.moveTo(A.x, A.y);
          const rectW = rect.width;
          if (activeTool === "ray") {
            const dx = B.x - A.x,
              dy = B.y - A.y,
              k = (rectW - A.x) / (dx || 1e-6);
            ctx.lineTo(A.x + dx * k, A.y + dy * k);
          } else {
            ctx.lineTo(B.x, B.y);
          }
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = color;
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }

    // ---- pattern markers -------------------------------------------------
    for (const m of markers || []) {
      const pxTime = typeof m.time === "string" ? toLWTime(m.time) : (m.time as number);
      const price = getPriceForMarker(m);
      if (price == null) continue;

      const pt = xyAt(pxTime, price);
      if (!pt) continue;

      const size = 6;
      const side =
        m.side ??
        (m.position === "aboveBar" ? "above" : m.position === "belowBar" ? "below" : "inBar");
      const fill =
        m.color ||
        (side === "above" ? "#ef5350" : side === "below" ? "#26a69a" : "#60a5fa");

      ctx.beginPath();
      if (side === "above") {
        ctx.moveTo(pt.x, pt.y - size);
        ctx.lineTo(pt.x - size, pt.y + size);
        ctx.lineTo(pt.x + size, pt.y + size);
      } else if (side === "below") {
        ctx.moveTo(pt.x, pt.y + size);
        ctx.lineTo(pt.x - size, pt.y - size);
        ctx.lineTo(pt.x + size, pt.y - size);
      } else {
        ctx.moveTo(pt.x, pt.y - size);
        ctx.lineTo(pt.x + size, pt.y);
        ctx.lineTo(pt.x, pt.y + size);
        ctx.lineTo(pt.x - size, pt.y);
      }
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();

      if (DRAW_MARKER_LABELS && (m as any).label) {
        ctx.fillStyle = "#e5e7eb";
        ctx.font = "10px Inter, system-ui, sans-serif";
        ctx.fillText((m as any).label, pt.x + 8, pt.y + (side === "above" ? 0 : -2));
      }
    }
  }

  // ---- mouse handlers ----------------------------------------------------
  function onCanvasPointerDown(ev: React.MouseEvent<HTMLCanvasElement>) {
    ev.preventDefault();
    ev.stopPropagation();
    if (!activeTool || activeTool === "select") return;
    const rect = (ev.target as HTMLCanvasElement).getBoundingClientRect();
    const x = ev.clientX - rect.left,
      y = ev.clientY - rect.top;
    const chart = chartRef.current,
      s = seriesRef.current;
    if (!chart || !s) return;
    const t = chart.timeScale().coordinateToTime(x) as UTCTimestamp | null;
    const p = s.coordinateToPrice(y) as number | null;
    if (t == null || p == null) return;

    let a = { time: Number(t), price: p };
    if (magnet) a = snapAnchor(a.time, a.price);

    const next = [...draft, a];
    if (["trendline", "ray", "measure", "fib_retracement", "fib_extension", "rect"].includes(activeTool)) {
      if (next.length === 2) {
        const id = crypto.randomUUID();
        addDrawing({
          id,
          type: activeTool as any,
          anchors: next,
          props: { color: "#eab308", width: 1.6, style: "solid", visible: true, locked: false },
          meta: { createdAt: Date.now(), updatedAt: Date.now(), z: 0 },
        });
        setDraft([]);
        setHover(null);
      } else setDraft(next);
    } else if (activeTool === "hline" || activeTool === "text") {
      const id = crypto.randomUUID();
      addDrawing({
        id,
        type: activeTool as any,
        anchors: [a],
        props: {
          color: "#60a5fa",
          width: 1.6,
          style: "solid",
          visible: true,
          locked: false,
          label: activeTool === "text" ? "Label" : undefined,
        },
        meta: { createdAt: Date.now(), updatedAt: Date.now(), z: 0 },
      });
      setDraft([]);
      setHover(null);
    } else if (activeTool === "path") {
      setDraft(next);
    }
    scheduleDraw();
  }

  function onCanvasPointerMove(ev: React.MouseEvent<HTMLCanvasElement>) {
    if (!activeTool || activeTool === "select") return;
    const rect = (ev.target as HTMLCanvasElement).getBoundingClientRect();
    const x = ev.clientX - rect.left,
      y = ev.clientY - rect.top;
    const chart = chartRef.current,
      s = seriesRef.current;
    if (!chart || !s) return;
    const t = chart.timeScale().coordinateToTime(x) as UTCTimestamp | null;
    const p = s.coordinateToPrice(y) as number | null;
    if (t == null || p == null) return;

    let a = { time: Number(t), price: p };
    if (magnet) a = snapAnchor(a.time, a.price);
    setHover(a);
    scheduleDraw();
  }

  function onCanvasDoubleClick() {
    if (activeTool === "path" && draft.length > 1) {
      const id = crypto.randomUUID();
      addDrawing({
        id,
        type: "path",
        anchors: hover ? [...draft, hover] : draft,
        props: { color: "#34d399", width: 1.6, style: "solid", visible: true, locked: false },
        meta: { createdAt: Date.now(), updatedAt: Date.now(), z: 0 },
      });
      setDraft([]);
      setHover(null);
      scheduleDraw();
    }
  }

  function onCanvasContextMenu(ev: React.MouseEvent<HTMLCanvasElement>) {
    ev.preventDefault();
    if (draft.length) {
      setDraft([]);
      setHover(null);
      scheduleDraw();
    }
  }

  return (
    <div ref={containerRef} style={{ position: "relative", width: "100%", height: 540 }}>
      <canvas
        ref={overlayRef}
        onMouseDown={onCanvasPointerDown}
        onMouseMove={onCanvasPointerMove}
        onDoubleClick={onCanvasDoubleClick}
        onContextMenu={onCanvasContextMenu}
        style={{ position: "absolute", left: 0, top: 0, width: "100%", height: "100%", zIndex: 5 }}
      />
    </div>
  );
}
