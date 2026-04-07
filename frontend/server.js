// Copyright F5, Inc. 2026
// Licensed under the MIT License. See LICENSE.
// server.js
import express from "express";

// If you deploy with different backends, set BACKEND_URL in env (e.g. http://127.0.0.1:8000)
const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";
const PORT = process.env.PORT || 3000;

const app = express();
app.use(express.static("public"));
app.use(express.json());

// --- Proxy: chat ask -> FastAPI /rag (FORWARDS AUTHORIZATION) ---
app.post("/ask", async (req, res) => {
  try {
    const auth = req.headers["authorization"]; // <- capture bearer
    const resp = await fetch(`${BACKEND_URL}/rag`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(auth ? { Authorization: auth } : {}), // <- forward it if present
      },
      body: JSON.stringify(req.body),
    });
    const data = await resp.json();
    res.status(resp.status).json(data);
  } catch (err) {
    console.error("Proxy /ask error:", err);
    res.status(500).json({ error: "Failed to reach backend." });
  }
});

// --- Proxy: token issuance (unprotected) ---
app.post("/token", async (req, res) => {
  try {
    const resp = await fetch(`${BACKEND_URL}/token`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    const data = await resp.json();
    res.status(resp.status).json(data);
  } catch (e) {
    console.error("Proxy /token error:", e);
    res.status(500).json({ error: "Token request failed" });
  }
});

// --- Optional: proxy /reload if you call it from the UI ---
app.post("/reload", async (req, res) => {
  try {
    const auth = req.headers["authorization"];
    const resp = await fetch(`${BACKEND_URL}/reload`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(auth ? { Authorization: auth } : {}),
      },
    });
    const data = await resp.json();
    res.status(resp.status).json(data);
  } catch (e) {
    console.error("Proxy /reload error:", e);
    res.status(500).json({ error: "Reload request failed" });
  }
});

app.get("/health", (_, res) => res.json({ ok: true, backend: BACKEND_URL }));

app.listen(PORT, () => {
  console.log(`Frontend running at http://localhost:${PORT}`);
  console.log(`Proxying to backend at ${BACKEND_URL}`);
});
