import { Streamlit, RenderData, withStreamlitConnection } from "streamlit-component-lib";
import { Chart } from "./Chart";
import { Toolbar, ToolbarState } from "./Toolbar";
import type { Drawing, PayloadIn, PayloadOut } from "./types";
import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

function Root(){
  const [args, setArgs] = useState<PayloadIn | null>(null);
  const [activeTool, setActiveTool] = useState<string | null>(null);
  const [drawings, setDrawings] = useState<Record<string, Drawing>>({});
  const [toolbar, setToolbar] = useState<ToolbarState>({ mode: "docked-right", x:12, y:12 });

  useEffect(()=>{
    function onRender(ev: Event){
      const d = (ev as CustomEvent<RenderData>).detail;
      const a = d.args as unknown as PayloadIn;
      setArgs(a);
      const key = `drawings:${a.symbol}@${a.timeframe}`;
      try{
        const local = localStorage.getItem(key);
        setDrawings(local ? JSON.parse(local) : (a.initial_drawings || {}));
      }catch{ setDrawings(a.initial_drawings || {}); }
      try{
        const ts = localStorage.getItem(`toolbar:${a.symbol}@${a.timeframe}`);
        setToolbar(ts ? JSON.parse(ts) : { mode: a.toolbar_default as any, x:12, y:12 });
      }catch{}
      Streamlit.setFrameHeight();
    }
    // @ts-ignore
    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
    Streamlit.setComponentReady();
    Streamlit.setFrameHeight();
    return ()=>{
      // @ts-ignore
      Streamlit.events.removeEventListener(Streamlit.RENDER_EVENT, onRender);
    }
  },[]);

  useEffect(()=>{
    if(!args) return;
    const out: PayloadOut = { drawings, toolbar };
    try{
      localStorage.setItem(`toolbar:${args.symbol}@${args.timeframe}`, JSON.stringify(toolbar));
    }catch{}
    Streamlit.setComponentValue(out as any);
  }, [drawings, toolbar, args]);

  if(!args) return null;

  const key = `${args.symbol}@${args.timeframe}`;

  return (
    <div style={{position:"relative"}}>
      <Chart
        data={args.ohlcv}
        drawings={drawings}
        setDrawings={setDrawings}
        activeTool={activeTool}
        magnet={args.magnet}
        toolbarKey={key}
      />
      <Toolbar
        state={toolbar}
        setState={setToolbar}
        active={activeTool as any}
        setActive={setActiveTool as any}
        onCmd={(cmd)=>{
          if(cmd==="clearAll"){ setDrawings({}); }
          else if(cmd==="hideAll"){
            const c: Record<string, Drawing> = {}; for(const [id,d] of Object.entries(drawings))
              c[id] = {...d, props:{...d.props, visible:false}}; setDrawings(c);
          } else if(cmd==="showAll"){
            const c: Record<string, Drawing> = {}; for(const [id,d] of Object.entries(drawings))
              c[id] = {...d, props:{...d.props, visible:true}}; setDrawings(c);
          }
          // undo/redo & selection can be added later
        }}
      />
    </div>
  );
}

const root = createRoot(document.body.appendChild(document.createElement("div")));
root.render(<Root/>);
