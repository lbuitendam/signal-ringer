import { useEffect, useRef, useState } from "react";
import type { DrawingType } from "./types";
import React from "react";


type Mode = "floating" | "docked-left" | "docked-right";
export interface ToolbarState { mode: Mode; x: number; y: number }
export interface ToolbarProps {
  state: ToolbarState; setState: (s: ToolbarState)=>void;
  active: DrawingType | null; setActive: (t: DrawingType | null)=>void;
  onCmd: (cmd: string)=>void;
}

const BTN: Array<{k: DrawingType | "select"; label: string}> = [
  {k:"select" as any, label:"Select"},
  {k:"trendline", label:"Trend"},
  {k:"ray", label:"Ray"},
  {k:"hline", label:"HLine"},
  {k:"rect", label:"Rect"},
  {k:"path", label:"Path"},
  {k:"text", label:"Text"},
  {k:"measure", label:"Measure"},
  {k:"fib_retracement", label:"FibRet"},
  {k:"fib_extension", label:"FibExt"}
];

export function Toolbar({state,setState,active,setActive,onCmd}:ToolbarProps){
  const ref = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<{ox:number; oy:number} | null>(null);

  useEffect(()=>{ // Dock positioning
    if(state.mode==="docked-left"){ setState({...state,x:12,y:12}); }
    if(state.mode==="docked-right"){ setState({...state,x:undefined as any,y:12}); }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[state.mode]);

  function onPointerDown(e: React.PointerEvent){
    if(state.mode!=="floating") return;
    const el = ref.current; if(!el) return;
    el.setPointerCapture(e.pointerId);
    setDrag({ox:e.clientX - (state.x||12), oy:e.clientY - (state.y||12)});
  }
  function onPointerMove(e: React.PointerEvent){
    if(!drag || state.mode!=="floating") return;
    setState({...state, x: e.clientX - drag.ox, y: e.clientY - drag.oy});
  }
  function onPointerUp(e: React.PointerEvent){
    const el = ref.current; if(!el) return;
    try{ el.releasePointerCapture(e.pointerId);}catch{}
    setDrag(null);
  }

  return (
    <div
      ref={ref}
      style={{
        position: "absolute",
        top: state.mode==="floating" ? (state.y ?? 12) : 12,
        left: state.mode==="docked-left" ? 12 : (state.mode==="floating" ? (state.x ?? 12) : "auto"),
        right: state.mode==="docked-right" ? 12 : "auto",
        background: "rgba(18,20,24,0.9)",
        border: "1px solid #333",
        borderRadius: 8,
        padding: 8,
        display: "flex",
        gap: 6,
        flexWrap: "wrap",
        zIndex: 9999,
        userSelect: "none",
        cursor: state.mode==="floating" ? "grab" : "default"
      }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      {BTN.map(b => (
        <button
          key={b.k as string}
          onClick={()=> setActive(b.k as DrawingType)}
          style={{
            padding: "4px 6px",
            background: (active===b.k ? "#3b82f6" : "#1f2937"),
            color: "#e5e7eb", border: "1px solid #374151",
            borderRadius: 6, fontSize: 12
          }}
          title={b.label}
        >{b.label}</button>
      ))}
      <div style={{display:"flex",gap:6, marginLeft:6}}>
        <button onClick={()=>onCmd("undo")}>Undo</button>
        <button onClick={()=>onCmd("redo")}>Redo</button>
        <button onClick={()=>onCmd("hideAll")}>Hide</button>
        <button onClick={()=>onCmd("showAll")}>Show</button>
        <button onClick={()=>onCmd("deleteSelected")}>Delete</button>
        <button onClick={()=>onCmd("clearAll")}>Clear</button>
        <select
          value={state.mode}
          onChange={(e)=>setState({...state, mode: e.target.value as Mode})}
          style={{background:"#111827", color:"#e5e7eb", border:"1px solid #374151", borderRadius:6}}
          title="Dock / Float"
        >
          <option value="floating">Floating</option>
          <option value="docked-left">Dock Left</option>
          <option value="docked-right">Dock Right</option>
        </select>
      </div>
    </div>
  );
}
