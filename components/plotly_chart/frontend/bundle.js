(function () {
  const { Streamlit } = window.streamlitComponentLib || {};
  if (!Streamlit) { console.error("plotly_chart: Streamlit lib missing"); return; }
  try { Streamlit.setComponentReady(); Streamlit.setFrameHeight(640); } catch (_) {}

  const root = document.getElementById("root");
  const chartHost = document.getElementById("chart");
  if (!root || !chartHost) { console.error("plotly_chart: DOM missing"); return; }

  let gd = null;
  let lastUid = null;
  let initialShapesSet = false;

  function ensurePlot(args) {
    const { uid, candles, shapes, layout, config, height } = args;
    lastUid = uid;

    const trace = {
      type: "candlestick",
      x: candles?.x || [],
      open: candles?.open || [],
      high: candles?.high || [],
      low: candles?.low || [],
      close: candles?.close || [],
      increasing: { line: { color: "#00e5ff" } },
      decreasing: { line: { color: "#ff4d4f" } },
    };

    const ly = Object.assign({
      dragmode: "zoom",
      autosize: true,
      margin: { l: 40, r: 16, t: 20, b: 24 },
      paper_bgcolor: "#0d1117",
      plot_bgcolor: "#0d1117",
      xaxis: { rangeslider: { visible: false }, gridcolor: "rgba(255,255,255,0.10)", zeroline: false },
      yaxis: { gridcolor: "rgba(255,255,255,0.10)", zeroline: false },
      font: { color: "#c9d1d9" },
      height: height || 640,
    }, layout || {});

    if (Array.isArray(shapes)) {
      // apply server shapes on (re)render
      ly.shapes = shapes;
      initialShapesSet = true;
    }

    const cfg = Object.assign({
      displaylogo: false,
      responsive: true,
      modeBarButtonsToAdd: [
        "drawline","drawopenpath","drawrect","drawcircle","drawclosedpath","eraseshape","toggleSpikelines"
      ],
      scrollZoom: true,
    }, config || {});

    if (!gd) {
      gd = chartHost;
      Plotly.newPlot(gd, [trace], ly, cfg).then(() => {
        attachHandlers();
        sizeToHost(ly.height);
      });
    } else {
      Plotly.react(gd, [trace], ly, cfg).then(() => {
        sizeToHost(ly.height);
      });
    }
  }

  function sizeToHost(h) {
    try {
      const width = root.clientWidth || gd.clientWidth || 800;
      const height = root.clientHeight || h || 640;
      Plotly.relayout(gd, { width, height });
    } catch (_) {}
    Streamlit.setFrameHeight(root.clientHeight || h || 640);
  }

  function attachHandlers() {
    if (!gd) return;

    // Receive tool changes from the toolbar iframe via parent
    window.addEventListener("message", (ev) => {
      const msg = ev.data || {};
      if (!msg || msg.channel !== `proplotly:${lastUid}`) return;
      if (msg.type === "tool") {
        const tool = msg.tool;
        // Plotly supports setting dragmode to the draw tools directly
        if (tool === "pan" || tool === "zoom" || tool.startsWith("draw") || tool === "eraseshape" || tool === "lasso" || tool === "select") {
          Plotly.relayout(gd, { dragmode: tool });
        } else if (tool === "clearshapes") {
          Plotly.relayout(gd, { shapes: [] });
          Streamlit.setComponentValue({ dirty: true, shapes: [] });
        }
      }
    });

    // Persist shapes whenever user draws/edits/erases
    gd.on("plotly_relayout", () => {
      const shapes = Array.isArray(gd.layout.shapes) ? gd.layout.shapes : [];
      Streamlit.setComponentValue({ dirty: true, shapes });
    });

    window.addEventListener("resize", () => sizeToHost(gd?.layout?.height || 640));
  }

  function render(args) { ensurePlot(args || {}); }

  function onRender(event) {
    const data = event.detail;
    if (!data) return;
    render(data.args || {});
  }

  const RENDER_EVENT = (typeof Streamlit.RENDER_EVENT === "string") ? Streamlit.RENDER_EVENT : "render";
  Streamlit.events.addEventListener(RENDER_EVENT, onRender);
  Streamlit.setComponentReady();
  Streamlit.setFrameHeight(640);
})();
