import React from "react";

export type ToolName =
  | "select"
  | "trendline"
  | "ray"
  | "rect"
  | "hline"
  | "path"
  | "text"
  | "measure"
  | "fib_retracement"
  | "fib_extension";

type Dock = "left" | "right" | "floating";

export interface ToolbarProps {
  activeTool: ToolName | null;
  onSelect: (tool: ToolName | null) => void;
  dock: Dock;
  setDock: (d: Dock) => void;
  canClear?: boolean;
  onClear?: () => void;
  magnet?: boolean;           // optional indicator
}

const tools: { key: ToolName; label: string }[] = [
  { key: "select",          label: "Sel" },
  { key: "trendline",       label: "TL" },
  { key: "ray",             label: "Ray" },
  { key: "rect",            label: "Rect" },
  { key: "hline",           label: "H" },
  { key: "path",            label: "Path" },
  { key: "text",            label: "Txt" },
  { key: "measure",         label: "Meas" },
  { key: "fib_retracement", label: "FibR" },
  { key: "fib_extension",   label: "FibX" },
];

export const Toolbar: React.FC<ToolbarProps> = ({
  activeTool,
  onSelect,
  dock,
  setDock,
  canClear,
  onClear,
  magnet,
}) => {
  const base: React.CSSProperties = {
    position: "absolute",
    top: 8,
    zIndex: 3,
    display: "flex",
    flexDirection: "column",
    gap: 6,
    background: "#0b0f15",
    border: "1px solid #222",
    borderRadius: 10,
    padding: 8,
    boxShadow: "0 8px 24px rgba(0,0,0,0.35)",
    userSelect: "none",
  };

  const pos: React.CSSProperties =
    dock === "left"
      ? { left: 8 }
      : dock === "right"
      ? { right: 8 }
      : { left: 16, top: 16, cursor: "move" }; // (floating = manual drag if you add it later)

  const btn: React.CSSProperties = {
    font: "12px/1 system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
    color: "#e5e7eb",
    background: "transparent",
    border: "1px solid #2a2f36",
    borderRadius: 8,
    padding: "6px 8px",
    cursor: "pointer",
  };

  const btnActive: React.CSSProperties = {
    ...btn,
    background: "#1e293b",
    borderColor: "#3b4251",
  };

  const row: React.CSSProperties = { display: "grid", gridTemplateColumns: "repeat(5, auto)", gap: 6 };

  return (
    <div style={{ ...base, ...pos }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 6 }}>
        <button
          style={btn}
          onClick={() => setDock(dock === "left" ? "right" : "left")}
          title={`Dock: ${dock}`}
        >
          {dock === "left" ? "⟷ Right" : "⟷ Left"}
        </button>
        {typeof magnet === "boolean" && (
          <div
            title={magnet ? "Magnet: on" : "Magnet: off"}
            style={{
              ...btn,
              padding: "6px 10px",
              background: magnet ? "#0b3d2a" : undefined,
              borderColor: magnet ? "#115e3e" : "#2a2f36",
            }}
          >
            🧲
          </div>
        )}
      </div>

      <div style={row}>
        {tools.map((t) => (
          <button
            key={t.key}
            style={activeTool === t.key ? btnActive : btn}
            onClick={() => onSelect(activeTool === t.key ? null : t.key)}
            title={t.label}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
        <button
          style={{ ...btn, color: "#fca5a5", borderColor: "#4c1d1d" }}
          onClick={() => onSelect(null)}
          title="Stop drawing"
        >
          ✕ Stop
        </button>
        <button
          style={{ ...btn, color: "#f87171", borderColor: "#4c1d1d", opacity: canClear ? 1 : 0.5 }}
          onClick={() => canClear && onClear && onClear()}
          disabled={!canClear}
          title="Clear all drawings"
        >
          Clear
        </button>
      </div>
    </div>
  );
};

export default Toolbar;
