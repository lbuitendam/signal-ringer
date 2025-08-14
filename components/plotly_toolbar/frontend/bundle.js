(function () {
  const { Streamlit } = window.streamlitComponentLib || {};
  if (!Streamlit) { console.error("plotly_toolbar: Streamlit lib missing"); return; }
  try { Streamlit.setComponentReady(); Streamlit.setFrameHeight(64); } catch (_) {}

  const root = document.getElementById("root");
  const toolbox = document.getElementById("toolbox");
  if (!root || !toolbox) { console.error("plotly_toolbar: DOM missing"); return; }

  let UID = null;

  function render(args) {
    UID = args?.uid || "default";
    // Hook up clicks
    toolbox.addEventListener("click", (e) => {
      const btn = e.target.closest(".btn");
      if (!btn) return;
      const tool = btn.dataset.tool;

      // Visual state
      [...toolbox.querySelectorAll(".btn")].forEach(b => b.classList.toggle("active", b === btn));

      // Broadcast to parent; chart iframe listens for this channel
      window.parent.postMessage({ channel: `proplotly:${UID}`, type: "tool", tool }, "*");
      // Do NOT call setComponentValue here -> no Streamlit rerun
    }, { once: true }); // prevent duplicate listeners on rerender

    Streamlit.setFrameHeight(root.clientHeight || 64);
  }

  function onRender(event) {
    const data = event.detail;
    if (!data) return;
    render(data.args || {});
  }

  const RENDER_EVENT = (typeof Streamlit.RENDER_EVENT === "string") ? Streamlit.RENDER_EVENT : "render";
  Streamlit.events.addEventListener(RENDER_EVENT, onRender);
  Streamlit.setComponentReady();
  Streamlit.setFrameHeight(64);
})();
