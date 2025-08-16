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
  onReady?: (api: IChartApi) => void;

  markers?: Marker[];
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
  const seriesRef = useRef<any>(null);
  const normDataRef = useRef<CandlestickData[]>([]);
  const overlaySeriesMap = useRef<Map<string, ISeriesApi<"Line">>>(new Map());

  const [draft, setDraft] = useState<Anchor[]>([]);

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

    // overlay canvas
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
      drawAll();
    });
    ro.observe(el);

    const onRangeChange = () => {
      sizeOverlay();
      drawAll();
    };
    chart.timeScale().subscribeVisibleTimeRangeChange(onRangeChange);
    const tsAny = chart.timeScale() as any;
    const hasLogical =
      typeof tsAny.subscribeVisibleLogicalRangeChange === "function" &&
      typeof tsAny.unsubscribeVisibleLogicalRangeChange === "function";
    if (hasLogical) tsAny.subscribeVisibleLogicalRangeChange(onRangeChange);

    return () => {
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

  useEffect(() => {
    const s = seriesRef.current;
    if (!s) return;
    const norm = normalizeData(data);
    if (!norm.length) return;
    normDataRef.current = norm;
    s.setData(norm);
    drawAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    // remove missing
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
        line.applyOptions({
          color: o.color || "#33cccc",
          lineWidth: lw as any,
          lineStyle: ls,
        } as any);
      }

      const lineData = (o.data || [])
        .filter((r) => r && r.time != null && Number.isFinite(r.value))
        .map((r) => ({ time: toLWTime(r.time) as UTCTimestamp, value: +r.value }))
        .sort((a, b) => (a.time as number) - (b.time as number));
      if (lineData.length) line.setData(lineData);
    }

    drawAll();
  }, [overlays]);

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

  function toXY(a: Anchor) {
    const chart = chartRef.current;
    const s = seriesRef.current;
    if (!chart || !s) return null;
    const x = chart.timeScale().timeToCoordinate(unix(a.time));
    if (x == null) return null;
    const y = s.priceToCoordinate(a.price) ?? null;
    if (y == null) return null;
    return { x, y };
  }

  function nearestCandle(time: number, price: number): Anchor {
    const arr = normDataRef.current;
    if (!arr.length) return { time, price };
    let idx = 0,
      best = Number.POSITIVE_INFINITY;
    for (let i = 0; i < arr.length; i++) {
      const diff = Math.abs((arr[i].time as number) - time);
      if (diff < best) {
        best = diff;
        idx = i;
      }
    }
    return { time: arr[idx].time as number, price };
  }

  function addDrawing(d: Drawing) {
    const copy = { ...drawings, [d.id]: d };
    setDrawings(copy);
    try {
      localStorage.setItem(`drawings:${toolbarKey}`, JSON.stringify(copy));
    } catch {}
  }

  // helpers for markers
  function findCandleAt(ts: number): CandlestickData | null {
    const arr = normDataRef.current;
    if (!arr.length) return null;
    // binary search would be nicer; linear is fine for now
    for (let i = 0; i < arr.length; i++) {
      if ((arr[i].time as number) === ts) return arr[i];
    }
    return null;
  }

  function normalizeMarker(m: Marker): { x: number | null; y: number | null; side: "above" | "below" | "inBar"; color: string; label?: string } {
    const chart = chartRef.current;
    const s = seriesRef.current;
    if (!chart || !s) return { x: null, y: null, side: "inBar", color: "#60a5fa" };

    const ts = toLWTime(m.time);
    const candle = findCandleAt(ts);
    let side: "above" | "below" | "inBar" =
      m.side || (m.position === "aboveBar" ? "above" : m.position === "belowBar" ? "below" : "inBar");

    // pick a price if absent
    let price =
      typeof m.price === "number"
        ? m.price
        : candle
        ? side === "above"
          ? (candle.high as number)
          : side === "below"
          ? (candle.low as number)
          : (candle.close as number)
        : NaN;

    const x = chart.timeScale().timeToCoordinate(ts as UTCTimestamp);
    const y = s.priceToCoordinate(price);

    const color =
      m.color || (side === "above" ? "#ef5350" : side === "below" ? "#26a69a" : "#60a5fa");

    return { x: x ?? null, y: y ?? null, side, color, label: m.label };
  }

  function drawAll() {
    const canvas = overlayRef.current,
      chart = chartRef.current,
      s = seriesRef.current;
    if (!canvas || !chart || !s) return;
    const ctx = canvas.getContext("2d")!,
      rect = canvas.getBoundingClientRect();
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
    }

    // markers
    for (const raw of markers) {
      const m = normalizeMarker(raw);
      if (m.x == null || m.y == null) continue;
      const size = 6;
      ctx.beginPath();
      if (m.side === "above") {
        ctx.moveTo(m.x, m.y - size);
        ctx.lineTo(m.x - size, m.y + size);
        ctx.lineTo(m.x + size, m.y + size);
      } else if (m.side === "below") {
        ctx.moveTo(m.x, m.y + size);
        ctx.lineTo(m.x - size, m.y - size);
        ctx.lineTo(m.x + size, m.y - size);
      } else {
        // small diamond for inBar
        ctx.moveTo(m.x, m.y - size);
        ctx.lineTo(m.x + size, m.y);
        ctx.lineTo(m.x, m.y + size);
        ctx.lineTo(m.x - size, m.y);
      }
      ctx.closePath();
      ctx.fillStyle = m.color;
      ctx.fill();

      if (m.label) {
        ctx.fillStyle = "#e5e7eb";
        ctx.font = "10px Inter, system-ui, sans-serif";
        ctx.fillText(m.label, m.x + 8, m.y + (m.side === "above" ? 0 : -2));
      }
    }
  }

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
    if (magnet) a = nearestCandle(a.time, a.price);

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
    } else if (activeTool === "path") {
      setDraft(next);
    }
  }

  function onCanvasDoubleClick() {
    if (activeTool === "path" && draft.length > 1) {
      const id = crypto.randomUUID();
      addDrawing({
        id,
        type: "path",
        anchors: draft,
        props: { color: "#34d399", width: 1.6, style: "solid", visible: true, locked: false },
        meta: { createdAt: Date.now(), updatedAt: Date.now(), z: 0 },
      });
      setDraft([]);
    }
  }

  return (
    <div ref={containerRef} style={{ position: "relative", width: "100%", height: 540 }}>
      <canvas
        ref={overlayRef}
        onMouseDown={onCanvasPointerDown}
        onDoubleClick={onCanvasDoubleClick}
        style={{ position: "absolute", left: 0, top: 0, width: "100%", height: "100%", zIndex: 5, pointerEvents: "none" }}
      />
    </div>
  );
}
