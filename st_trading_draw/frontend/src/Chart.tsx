import React, { useEffect, useRef, useState } from "react";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type UTCTimestamp,
} from "lightweight-charts";
import type { Anchor, Drawing, Ohlcv } from "./types";
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
}

export function Chart({
  data,
  drawings,
  setDrawings,
  activeTool,
  magnet,
  toolbarKey,
}: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [draft, setDraft] = useState<Anchor[]>([]);

  useEffect(() => {
    const el = containerRef.current!;
    const chart = createChart(el, {
      width: el.clientWidth,
      height: 520,
      layout: { background: { color: "#0e1117" }, textColor: "#e5e7eb" },
      timeScale: { rightOffset: 2, borderColor: "#333" },
      grid: { vertLines: { color: "#1f2937" }, horzLines: { color: "#1f2937" } },
    });

    const s = chart.addCandlestickSeries({
      upColor: "#26a69a",
      downColor: "#ef5350",
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
      borderVisible: false,
    });

    s.setData(
      data.map((d) => ({
        time: unix(d.time),
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      })) as CandlestickData[]
    );

    const ro = new ResizeObserver(() => {
      chart.applyOptions({ width: el.clientWidth });
      drawAll();
    });
    ro.observe(el);

    chart.timeScale().subscribeVisibleTimeRangeChange(drawAll);
    chart.timeScale().subscribeVisibleLogicalRangeChange?.(drawAll);

    chartRef.current = chart;
    seriesRef.current = s;

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    drawAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawings]);

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
    let idx = 0;
    let best = Number.POSITIVE_INFINITY;
    for (let i = 0; i < data.length; i++) {
      const diff = Math.abs(data[i].time - time);
      if (diff < best) {
        best = diff;
        idx = i;
      }
    }
    return { time: data[idx].time, price };
  }

  function addDrawing(d: Drawing) {
    const copy = { ...drawings, [d.id]: d };
    setDrawings(copy);
    persist(copy);
  }
  function updateDrawing(id: string, d: Partial<Drawing>) {
    const nd = { ...drawings[id], ...d, meta: { ...drawings[id].meta, updatedAt: Date.now() } };
    const copy = { ...drawings, [id]: nd };
    setDrawings(copy);
    persist(copy);
  }
  function persist(d: Record<string, Drawing>) {
    try {
      localStorage.setItem(`drawings:${toolbarKey}`, JSON.stringify(d));
    } catch {}
  }

  function drawAll() {
    const canvas = overlayRef.current;
    const chart = chartRef.current;
    const s = seriesRef.current;
    if (!canvas || !chart || !s) return;

    const ctx = canvas.getContext("2d")!;
    const { width, height } = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const items = Object.values(drawings).filter((d) => d.props.visible !== false);

    for (const d of items) {
      ctx.lineWidth = d.props.width ?? 1.5;
      ctx.strokeStyle = d.props.color ?? "#eab308";
      ctx.setLineDash(d.props.style === "dash" ? [6, 4] : d.props.style === "dot" ? [2, 4] : []);

      if (d.type === "trendline" || d.type === "ray") {
        const [a, b] = d.anchors;
        if (!a || !b) continue;
        const A = toXY(a);
        const B = toXY(b);
        if (!A || !B) continue;
        ctx.beginPath();
        ctx.moveTo(A.x, A.y);
        if (d.type === "ray") {
          const dx = B.x - A.x;
          const dy = B.y - A.y;
          const k = (width - A.x) / (dx || 1e-6);
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
        ctx.lineTo(width, y);
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
        const A = toXY(a);
        const B = toXY(b);
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
  }

  function onCanvasClick(ev: React.MouseEvent<HTMLCanvasElement>) {
    const rect = (ev.target as HTMLCanvasElement).getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const chart = chartRef.current;
    const s = seriesRef.current;
    if (!chart || !s) return;
    const t = chart.timeScale().coordinateToTime(x) as UTCTimestamp | null;
    const p = s.coordinateToPrice(y) as number | null;
    if (t == null || p == null) return;

    let a = { time: Number(t), price: p };
    if (magnet) a = nearestCandle(a.time, a.price);

    if (!activeTool || activeTool === "select") {
      setSelectedId(null);
      return;
    }

    const next = [...draft, a];

    if (
      activeTool === "trendline" ||
      activeTool === "ray" ||
      activeTool === "measure" ||
      activeTool === "fib_retracement" ||
      activeTool === "fib_extension" ||
      activeTool === "rect"
    ) {
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
      setDraft(next); // double-click to commit
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
    <div style={{ position: "relative", width: "100%", height: 540 }} ref={containerRef}>
      <canvas
        ref={overlayRef}
        onClick={onCanvasClick}
        onDoubleClick={onCanvasDoubleClick}
        style={{ position: "absolute", left: 0, top: 0, width: "100%", height: "100%", pointerEvents: "auto" }}
      />
    </div>
  );
}
