import React, { useEffect, useRef } from "react";
import { createChart, type IChartApi, type UTCTimestamp } from "lightweight-charts";
import type { PaneSpec } from "./types";

function toLWTime(t: number | string): number {
  if (typeof t === "number") return t > 1e12 ? Math.floor(t / 1000) : Math.floor(t);
  const ms = Date.parse(t);
  return Math.floor(ms / 1000);
}

type Props = {
  pane: PaneSpec;
  syncWith?: IChartApi | null;
};

const IndicatorsPane: React.FC<Props> = ({ pane, syncWith }) => {
  const divRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    const el = divRef.current;
    if (!el) return;
    if (!el.style.minHeight) el.style.minHeight = pane.height + "px";

    const chart = createChart(el, {
      width: el.clientWidth,
      height: pane.height,
      layout: { background: { color: "#0e1117" }, textColor: "#e5e7eb" },
      grid: { vertLines: { color: "#1f2937" }, horzLines: { color: "#1f2937" } },
      timeScale: { rightOffset: 2, borderVisible: false },
      rightPriceScale: {
        autoScale: !pane.yRange,
        scaleMargins: { top: 0.15, bottom: 0.15 },
      },
    });

    if (pane.yRange) {
      // If you need fixed range, do it after data with autoscale off.
      chart.priceScale("right").applyOptions({ autoScale: false, invertScale: false });
    }

    // lines
    for (const line of pane.lines || []) {
      const lw = Math.max(1, Math.round((line.width ?? 2) as number)) as any; // v5 branded type
      const ls = (line.dash === "dash" ? 1 : line.dash === "dot" ? 2 : 0) as any;

      const s = chart.addLineSeries({
        color: line.color || "#9ca3af",
        lineWidth: lw,
        lineStyle: ls,
        priceLineVisible: false,
      } as any);
      s.setData(
        (line.data || [])
          .filter((r) => r && r.time != null && Number.isFinite(r.value))
          .map((r) => ({ time: toLWTime(r.time) as UTCTimestamp, value: +r.value }))
          .sort((a, b) => (a.time as number) - (b.time as number))
      );
    }

    // histogram (MACD)
    for (const h of pane.hist || []) {
      const s = chart.addHistogramSeries({
        color: h.color || "#60a5fa",
        priceLineVisible: false,
      });
      s.setData(
        (h.data || [])
          .filter((r) => r && r.time != null && Number.isFinite(r.value))
          .map((r) => ({ time: toLWTime(r.time) as UTCTimestamp, value: +r.value }))
          .sort((a, b) => (a.time as number) - (b.time as number))
      );
    }

    // horizontal guide lines (optional if your build supports it)
    for (const hl of pane.hlines || []) {
      (chart as any).addHorizontalLine?.({
        price: hl.y,
        color: hl.color || "#555",
        lineStyle: (hl.dash === "dash" ? 1 : hl.dash === "dot" ? 2 : 0) as any,
        lineWidth: 1 as any,
      });
    }

    chartRef.current = chart;

    const ro = new ResizeObserver(() => {
      chart.applyOptions({ width: el.clientWidth, height: pane.height });
    });
    ro.observe(el);

    // one-way sync from main chart
    if (syncWith) {
      const sync = () => {
        const r = syncWith.timeScale().getVisibleRange();
        if (r) chart.timeScale().setVisibleRange(r);
      };
      // subscribe with the handler itself, and unsubscribe with the SAME handler
      syncWith.timeScale().subscribeVisibleTimeRangeChange(sync);
      sync(); // initial align

      return () => {
        ro.disconnect();
        syncWith.timeScale().unsubscribeVisibleTimeRangeChange(sync);
        chart.remove();
        chartRef.current = null;
      };
    }

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <div ref={divRef} style={{ width: "100%", height: pane.height }} />;
};

export default IndicatorsPane;
