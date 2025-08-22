// st_trading_draw/frontend/src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import Root from "./index"; // default export from your index.tsx (withStreamlitConnection)

const el = document.getElementById("root");
if (!el) {
  throw new Error("No #root element found in index.html");
}
ReactDOM.createRoot(el).render(<Root />);