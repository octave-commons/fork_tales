from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import cycle
from typing import Any
from urllib.parse import parse_qs, urlparse

from .ai import _embed_text, _eta_mu_normalize_vector, _eta_mu_resize_vector


WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
WS_CLIENT_FRAME_MAX_BYTES = 1_048_576

DEFAULT_HOST = os.getenv("EMBED_BENCH_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("EMBED_BENCH_PORT", "8890") or "8890")

BENCH_TEXTS: tuple[str, ...] = (
    "Embeddings capture semantic intent and neighborhood structure.",
    "NPU acceleration can improve tail latency for local vector workloads.",
    "Matryoshka representation allows truncation at smaller dimensions.",
    "The witness thread records every benchmark turn as append-only evidence.",
    "Fast in-process embedding paths avoid network overhead and jitter.",
    "A stable benchmark reports p50 p90 p99 max and throughput.",
    "Small sentence prompts keep tokenizer costs bounded and repeatable.",
    "Semantic vectors help map user intent into field-space coordinates.",
    "Deterministic benchmark loops expose regressions before deployment.",
    "Latency spikes often correlate with model warmup and cache misses.",
)

HTML_INDEX = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Embedding Benchmark Stream</title>
  <style>
    :root {
      --bg-0: #071018;
      --bg-1: #102134;
      --bg-2: #17324a;
      --ink-0: #e8f2ff;
      --ink-1: #b8d2ec;
      --ink-2: #8ca9c6;
      --good: #53d0a8;
      --warn: #f1bf65;
      --bad: #ff8f91;
      --accent: #57b6ff;
      --panel: rgba(9, 21, 34, 0.85);
      --line: rgba(144, 182, 218, 0.24);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: "JetBrains Mono", "Fira Code", "Menlo", monospace;
      color: var(--ink-0);
      background:
        radial-gradient(1200px 520px at 18% -12%, #244666, transparent 58%),
        radial-gradient(1100px 700px at 86% -25%, #25563f, transparent 60%),
        linear-gradient(165deg, var(--bg-0) 0%, var(--bg-1) 55%, var(--bg-2) 100%);
      min-height: 100vh;
      padding: 18px;
    }

    .layout {
      width: min(1160px, 100%);
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 14px;
    }

    .card {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel);
      box-shadow: 0 14px 36px rgba(0, 0, 0, 0.28);
      overflow: hidden;
    }

    .card h2 {
      margin: 0;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--ink-1);
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
    }

    .card .body {
      padding: 12px 14px;
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }

    .kpi {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      background: rgba(7, 17, 26, 0.45);
    }

    .kpi .label {
      color: var(--ink-2);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }

    .kpi .value {
      font-size: 18px;
      font-weight: 700;
      line-height: 1.1;
    }

    .value.good { color: var(--good); }
    .value.warn { color: var(--warn); }
    .value.bad { color: var(--bad); }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 8px;
    }

    label {
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--ink-2);
      margin-bottom: 4px;
    }

    input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 9px;
      color: var(--ink-0);
      background: rgba(5, 13, 20, 0.6);
      padding: 8px 9px;
      font: inherit;
      font-size: 13px;
    }

    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }

    button {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 11px;
      color: var(--ink-0);
      background: rgba(5, 14, 23, 0.7);
      font: inherit;
      font-size: 13px;
      cursor: pointer;
      transition: background 0.18s ease;
    }

    button:hover {
      background: rgba(22, 44, 68, 0.85);
    }

    .hint {
      color: var(--ink-2);
      font-size: 11px;
      margin-top: 8px;
      line-height: 1.45;
    }

    #streamCanvas {
      width: 100%;
      height: 210px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(7, 18, 28, 0.65);
      display: block;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }

    th,
    td {
      border-bottom: 1px solid rgba(140, 169, 198, 0.18);
      padding: 6px 4px;
      text-align: left;
      white-space: nowrap;
    }

    th {
      color: var(--ink-2);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      position: sticky;
      top: 0;
      background: rgba(9, 21, 34, 0.95);
    }

    .table-wrap {
      max-height: 280px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      margin-top: 8px;
    }

    @media (max-width: 980px) {
      .layout {
        grid-template-columns: 1fr;
      }

      #streamCanvas {
        height: 170px;
      }
    }
  </style>
</head>
<body>
  <div class=\"layout\">
    <section class=\"card\">
      <h2>Controls</h2>
      <div class=\"body\">
        <div class=\"row\">
          <div>
            <label for=\"backend\">Backend</label>
            <input id=\"backend\" value=\"openvino\" />
          </div>
          <div>
            <label for=\"model\">Model (optional)</label>
            <input id=\"model\" value=\"nomic-embed-text\" />
          </div>
        </div>

        <div class=\"row\">
          <div>
            <label for=\"interval\">Interval ms</label>
            <input id=\"interval\" type=\"number\" min=\"0\" step=\"1\" value=\"120\" />
          </div>
          <div>
            <label for=\"targetDim\">Target dim</label>
            <input id=\"targetDim\" type=\"number\" min=\"0\" step=\"1\" value=\"128\" />
          </div>
        </div>

        <div class=\"row\">
          <div>
            <label for=\"windowSize\">Window size</label>
            <input id=\"windowSize\" type=\"number\" min=\"16\" step=\"1\" value=\"512\" />
          </div>
          <div>
            <label for=\"maxChars\">Max chars</label>
            <input id=\"maxChars\" type=\"number\" min=\"64\" step=\"1\" value=\"1600\" />
          </div>
        </div>

        <div class=\"actions\">
          <button id=\"startBtn\">Start / Apply</button>
          <button id=\"stopBtn\">Stop</button>
          <button id=\"resetBtn\">Reset Stats</button>
        </div>

        <div class=\"hint\">
          Live stream arrives over WebSocket <code>/ws</code>. Metrics update in real time.
        </div>
      </div>
    </section>

    <section class=\"card\">
      <h2>Status</h2>
      <div class=\"body\">
        <div class=\"kpi-grid\">
          <div class=\"kpi\"><div class=\"label\">Run state</div><div id=\"runState\" class=\"value\">-</div></div>
          <div class=\"kpi\"><div class=\"label\">WebSocket</div><div id=\"wsState\" class=\"value warn\">connecting</div></div>
          <div class=\"kpi\"><div class=\"label\">Samples</div><div id=\"samples\" class=\"value\">0</div></div>
          <div class=\"kpi\"><div class=\"label\">Success rate</div><div id=\"successRate\" class=\"value\">0%</div></div>
          <div class=\"kpi\"><div class=\"label\">Throughput</div><div id=\"throughput\" class=\"value\">0 rps</div></div>
          <div class=\"kpi\"><div class=\"label\">Last dim</div><div id=\"lastDim\" class=\"value\">0</div></div>
          <div class=\"kpi\"><div class=\"label\">p50 / p90</div><div id=\"p50p90\" class=\"value\">0 / 0 ms</div></div>
          <div class=\"kpi\"><div class=\"label\">p99 / max</div><div id=\"p99max\" class=\"value\">0 / 0 ms</div></div>
        </div>
      </div>
    </section>

    <section class=\"card\">
      <h2>Latency Stream</h2>
      <div class=\"body\">
        <canvas id=\"streamCanvas\" width=\"860\" height=\"210\"></canvas>
      </div>
    </section>

    <section class=\"card\">
      <h2>Recent Samples</h2>
      <div class=\"body\">
        <div class=\"table-wrap\">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Latency</th>
                <th>OK</th>
                <th>Dim</th>
                <th>Error</th>
              </tr>
            </thead>
            <tbody id=\"sampleBody\"></tbody>
          </table>
        </div>
      </div>
    </section>
  </div>

  <script>
    const state = {
      latencies: [],
      events: [],
      ws: null,
      reconnectTimer: null,
    };

    const nodes = {
      backend: document.getElementById("backend"),
      model: document.getElementById("model"),
      interval: document.getElementById("interval"),
      targetDim: document.getElementById("targetDim"),
      windowSize: document.getElementById("windowSize"),
      maxChars: document.getElementById("maxChars"),
      startBtn: document.getElementById("startBtn"),
      stopBtn: document.getElementById("stopBtn"),
      resetBtn: document.getElementById("resetBtn"),
      wsState: document.getElementById("wsState"),
      runState: document.getElementById("runState"),
      samples: document.getElementById("samples"),
      successRate: document.getElementById("successRate"),
      throughput: document.getElementById("throughput"),
      lastDim: document.getElementById("lastDim"),
      p50p90: document.getElementById("p50p90"),
      p99max: document.getElementById("p99max"),
      sampleBody: document.getElementById("sampleBody"),
      canvas: document.getElementById("streamCanvas"),
    };

    function fmt(value, digits = 2) {
      const num = Number(value || 0);
      return Number.isFinite(num) ? num.toFixed(digits) : "0.00";
    }

    function renderKpis(snapshot) {
      const stats = (snapshot && snapshot.stats) || {};
      const latency = (snapshot && snapshot.latency) || {};
      const running = Boolean(snapshot && snapshot.running);

      nodes.runState.textContent = running ? "running" : "stopped";
      nodes.runState.className = running ? "value good" : "value warn";
      nodes.samples.textContent = String(stats.total_samples || 0);
      nodes.successRate.textContent = `${fmt((stats.success_rate || 0) * 100, 1)}%`;
      nodes.throughput.textContent = `${fmt(stats.throughput_rps || 0, 2)} rps`;
      nodes.lastDim.textContent = String(stats.last_vector_dim || 0);
      nodes.p50p90.textContent = `${fmt(latency.p50_ms, 2)} / ${fmt(latency.p90_ms, 2)} ms`;
      nodes.p99max.textContent = `${fmt(latency.p99_ms, 2)} / ${fmt(latency.max_ms, 2)} ms`;

      if (snapshot && snapshot.config) {
        const cfg = snapshot.config;
        nodes.backend.value = cfg.backend || "";
        nodes.model.value = cfg.model || "";
        nodes.interval.value = String(cfg.interval_ms || 0);
        nodes.targetDim.value = String(cfg.target_dim || 0);
        nodes.windowSize.value = String(cfg.window_size || 0);
        nodes.maxChars.value = String(cfg.max_chars || 0);
      }

      if (Array.isArray(snapshot && snapshot.recent_latencies_ms)) {
        state.latencies = snapshot.recent_latencies_ms.slice(-220);
      }
      if (Array.isArray(snapshot && snapshot.recent_events)) {
        state.events = snapshot.recent_events.slice(-28);
      }
      drawLatencyChart();
      renderEvents();
    }

    function renderEvents() {
      const rows = state.events.slice(-28).reverse();
      nodes.sampleBody.innerHTML = rows
        .map((row) => {
          const ok = row.ok ? "yes" : "no";
          const err = row.error ? String(row.error).slice(0, 90) : "";
          return `<tr>
            <td>${row.ts || ""}</td>
            <td>${fmt(row.latency_ms, 2)} ms</td>
            <td>${ok}</td>
            <td>${row.vector_dim || 0}</td>
            <td>${err}</td>
          </tr>`;
        })
        .join("");
    }

    function pushSample(event) {
      if (!event) {
        return;
      }
      const latency = Number(event.latency_ms || 0);
      if (Number.isFinite(latency) && latency >= 0) {
        state.latencies.push(latency);
        if (state.latencies.length > 220) {
          state.latencies = state.latencies.slice(-220);
        }
      }
      state.events.push(event);
      if (state.events.length > 28) {
        state.events = state.events.slice(-28);
      }
      drawLatencyChart();
      renderEvents();
    }

    function drawLatencyChart() {
      const canvas = nodes.canvas;
      const ctx = canvas.getContext("2d");
      const data = state.latencies;
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "rgba(7, 18, 28, 0.65)";
      ctx.fillRect(0, 0, w, h);

      if (!data.length) {
        ctx.fillStyle = "#8ca9c6";
        ctx.font = "13px JetBrains Mono";
        ctx.fillText("Waiting for benchmark samples...", 14, 26);
        return;
      }

      const min = Math.min(...data);
      const max = Math.max(...data);
      const span = Math.max(0.001, max - min);
      const padX = 10;
      const padY = 12;
      const drawW = w - padX * 2;
      const drawH = h - padY * 2;

      ctx.strokeStyle = "rgba(140, 169, 198, 0.26)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 3; i += 1) {
        const y = padY + (drawH * i) / 3;
        ctx.beginPath();
        ctx.moveTo(padX, y);
        ctx.lineTo(w - padX, y);
        ctx.stroke();
      }

      ctx.strokeStyle = "#57b6ff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      data.forEach((value, index) => {
        const x = padX + (drawW * index) / Math.max(1, data.length - 1);
        const y = padY + drawH - ((value - min) / span) * drawH;
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      const latest = data[data.length - 1];
      ctx.fillStyle = "#e8f2ff";
      ctx.font = "12px JetBrains Mono";
      ctx.fillText(`latest ${fmt(latest, 2)} ms`, 14, 18);
      ctx.fillText(`min ${fmt(min, 2)} ms`, 14, h - 10);
      ctx.fillText(`max ${fmt(max, 2)} ms`, w - 110, h - 10);
    }

    async function postJson(path, body) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    }

    function setWsState(text, mode) {
      nodes.wsState.textContent = text;
      nodes.wsState.className = `value ${mode || "warn"}`;
    }

    function connectWs() {
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${window.location.host}/ws`);
      state.ws = ws;

      ws.onopen = () => {
        setWsState("connected", "good");
      };

      ws.onmessage = (event) => {
        let payload = null;
        try {
          payload = JSON.parse(event.data);
        } catch (_err) {
          return;
        }
        if (!payload || typeof payload !== "object") {
          return;
        }
        if ((payload.type === "snapshot" || payload.type === "state") && payload.state) {
          renderKpis(payload.state);
          return;
        }
        if (payload.type === "sample") {
          if (payload.event) {
            pushSample(payload.event);
          }
          if (payload.state) {
            renderKpis(payload.state);
          }
          return;
        }
      };

      ws.onclose = () => {
        setWsState("reconnecting", "warn");
        if (state.reconnectTimer) {
          clearTimeout(state.reconnectTimer);
        }
        state.reconnectTimer = setTimeout(connectWs, 1400);
      };

      ws.onerror = () => {
        setWsState("error", "bad");
      };
    }

    nodes.startBtn.addEventListener("click", async () => {
      try {
        const payload = {
          backend: nodes.backend.value,
          model: nodes.model.value,
          interval_ms: Number(nodes.interval.value),
          target_dim: Number(nodes.targetDim.value),
          window_size: Number(nodes.windowSize.value),
          max_chars: Number(nodes.maxChars.value),
          reset: true,
        };
        const response = await postJson("/api/control/start", payload);
        if (response && response.state) {
          renderKpis(response.state);
        }
      } catch (err) {
        setWsState(`start failed: ${String(err)}`, "bad");
      }
    });

    nodes.stopBtn.addEventListener("click", async () => {
      try {
        const response = await postJson("/api/control/stop", {});
        if (response && response.state) {
          renderKpis(response.state);
        }
      } catch (err) {
        setWsState(`stop failed: ${String(err)}`, "bad");
      }
    });

    nodes.resetBtn.addEventListener("click", async () => {
      try {
        const response = await postJson("/api/control/reset", {});
        if (response && response.state) {
          renderKpis(response.state);
        }
      } catch (err) {
        setWsState(`reset failed: ${String(err)}`, "bad");
      }
    });

    async function bootstrap() {
      try {
        const response = await fetch("/api/state", { cache: "no-store" });
        if (response.ok) {
          const snapshot = await response.json();
          renderKpis(snapshot);
        }
      } catch (_err) {
        setWsState("state fetch failed", "bad");
      }
      connectWs();
    }

    bootstrap();
  </script>
</body>
</html>
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_compact(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _clamp_int(value: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def _as_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    p = max(0.0, min(1.0, float(percentile)))
    index = (len(sorted_values) - 1) * p
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return float(sorted_values[lo])
    weight = index - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean_ms": 0.0,
            "stddev_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "min_ms": 0.0,
        }

    ordered = sorted(float(v) for v in values)
    count = float(len(ordered))
    mean = sum(ordered) / len(ordered)
    variance = sum((value - mean) ** 2 for value in ordered) / max(1, len(ordered))
    return {
        "count": count,
        "mean_ms": mean,
        "stddev_ms": math.sqrt(variance),
        "p50_ms": _percentile(ordered, 0.50),
        "p90_ms": _percentile(ordered, 0.90),
        "p99_ms": _percentile(ordered, 0.99),
        "max_ms": ordered[-1],
        "min_ms": ordered[0],
    }


@dataclass
class BenchConfig:
    backend: str = "openvino"
    model: str = ""
    interval_ms: int = 120
    window_size: int = 512
    target_dim: int = 128
    max_chars: int = 1600


class BenchmarkState:
    def __init__(self, config: BenchConfig):
        self._lock = threading.Lock()
        self._config = BenchConfig(**vars(config))

        self._running = False
        self._started_at_iso = ""
        self._last_sample_at_iso = ""
        self._active_started_perf: float | None = None
        self._elapsed_active_sec = 0.0

        self._total_samples = 0
        self._success_samples = 0
        self._error_samples = 0
        self._last_vector_dim = 0
        self._last_latency_ms = 0.0
        self._last_error = ""

        self._latencies = deque(maxlen=self._config.window_size)
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=220)

    def _reset_locked(self) -> None:
        self._started_at_iso = ""
        self._last_sample_at_iso = ""
        self._active_started_perf = None
        self._elapsed_active_sec = 0.0
        self._total_samples = 0
        self._success_samples = 0
        self._error_samples = 0
        self._last_vector_dim = 0
        self._last_latency_ms = 0.0
        self._last_error = ""
        self._latencies.clear()
        self._recent_events.clear()

    def update_config(
        self,
        *,
        backend: str | None = None,
        model: str | None = None,
        interval_ms: int | None = None,
        window_size: int | None = None,
        target_dim: int | None = None,
        max_chars: int | None = None,
    ) -> BenchConfig:
        with self._lock:
            if backend is not None:
                clean = str(backend).strip().lower()
                if clean:
                    self._config.backend = clean
            if model is not None:
                self._config.model = str(model).strip()
            if interval_ms is not None:
                self._config.interval_ms = max(0, int(interval_ms))
            if target_dim is not None:
                self._config.target_dim = max(0, int(target_dim))
            if max_chars is not None:
                self._config.max_chars = max(64, int(max_chars))

            if window_size is not None:
                next_window = max(16, int(window_size))
                if next_window != self._config.window_size:
                    prior = list(self._latencies)
                    self._latencies = deque(prior[-next_window:], maxlen=next_window)
                    self._config.window_size = next_window

            return BenchConfig(**vars(self._config))

    def start(self, *, reset: bool = False) -> dict[str, Any]:
        with self._lock:
            if reset:
                self._reset_locked()
            if not self._running:
                self._running = True
                self._active_started_perf = time.perf_counter()
                if not self._started_at_iso:
                    self._started_at_iso = _utc_now_iso()
            return self._snapshot_locked(include_recent=True)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._running:
                if self._active_started_perf is not None:
                    self._elapsed_active_sec += max(
                        0.0, time.perf_counter() - self._active_started_perf
                    )
                self._active_started_perf = None
                self._running = False
            return self._snapshot_locked(include_recent=True)

    def reset(self) -> dict[str, Any]:
        with self._lock:
            running = self._running
            self._reset_locked()
            if running:
                self._running = True
                self._active_started_perf = time.perf_counter()
                self._started_at_iso = _utc_now_iso()
            return self._snapshot_locked(include_recent=True)

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def current_config(self) -> BenchConfig:
        with self._lock:
            return BenchConfig(**vars(self._config))

    def record_sample(
        self,
        *,
        ok: bool,
        latency_ms: float,
        vector_dim: int,
        error: str,
        sample_text: str,
    ) -> dict[str, Any]:
        with self._lock:
            self._total_samples += 1
            self._last_sample_at_iso = _utc_now_iso()
            self._last_latency_ms = max(0.0, float(latency_ms))
            self._last_vector_dim = max(0, int(vector_dim))

            if ok:
                self._success_samples += 1
                self._latencies.append(self._last_latency_ms)
                self._last_error = ""
            else:
                self._error_samples += 1
                self._last_error = str(error).strip()[:220]

            event = {
                "seq": self._total_samples,
                "ts": self._last_sample_at_iso,
                "ok": bool(ok),
                "latency_ms": round(self._last_latency_ms, 4),
                "vector_dim": self._last_vector_dim,
                "error": self._last_error,
                "sample": sample_text[:120],
            }
            self._recent_events.append(event)
            return event

    def snapshot(self, *, include_recent: bool = False) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked(include_recent=include_recent)

    def _snapshot_locked(self, *, include_recent: bool) -> dict[str, Any]:
        elapsed = self._elapsed_active_sec
        if self._running and self._active_started_perf is not None:
            elapsed += max(0.0, time.perf_counter() - self._active_started_perf)

        total = max(0, self._total_samples)
        success = max(0, self._success_samples)
        errors = max(0, self._error_samples)
        success_rate = (success / total) if total > 0 else 0.0
        throughput = (success / elapsed) if elapsed > 0 else 0.0

        latencies = list(self._latencies)
        latency = _latency_summary(latencies)
        payload: dict[str, Any] = {
            "running": self._running,
            "config": {
                "backend": self._config.backend,
                "model": self._config.model,
                "interval_ms": self._config.interval_ms,
                "window_size": self._config.window_size,
                "target_dim": self._config.target_dim,
                "max_chars": self._config.max_chars,
            },
            "stats": {
                "started_at": self._started_at_iso,
                "last_sample_at": self._last_sample_at_iso,
                "elapsed_active_sec": round(elapsed, 3),
                "total_samples": total,
                "success_samples": success,
                "error_samples": errors,
                "success_rate": round(success_rate, 6),
                "throughput_rps": round(throughput, 6),
                "last_latency_ms": round(self._last_latency_ms, 4),
                "last_vector_dim": self._last_vector_dim,
                "last_error": self._last_error,
            },
            "latency": {
                "count": int(latency.get("count", 0.0)),
                "mean_ms": round(latency.get("mean_ms", 0.0), 6),
                "stddev_ms": round(latency.get("stddev_ms", 0.0), 6),
                "p50_ms": round(latency.get("p50_ms", 0.0), 6),
                "p90_ms": round(latency.get("p90_ms", 0.0), 6),
                "p99_ms": round(latency.get("p99_ms", 0.0), 6),
                "max_ms": round(latency.get("max_ms", 0.0), 6),
                "min_ms": round(latency.get("min_ms", 0.0), 6),
            },
            "ts": _utc_now_iso(),
        }

        if include_recent:
            payload["recent_latencies_ms"] = [
                round(value, 4) for value in latencies[-220:]
            ]
            payload["recent_events"] = list(self._recent_events)[-80:]
        return payload


class WebSocketHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: set[socket.socket] = set()

    def add(self, client: socket.socket) -> None:
        with self._lock:
            self._clients.add(client)

    def remove(self, client: socket.socket) -> None:
        with self._lock:
            self._clients.discard(client)

    def count(self) -> int:
        with self._lock:
            return len(self._clients)

    def broadcast(self, payload: dict[str, Any]) -> None:
        frame = websocket_frame_text(_json_compact(payload))
        stale: list[socket.socket] = []
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.sendall(frame)
            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
                OSError,
            ):
                stale.append(client)
        if stale:
            with self._lock:
                for client in stale:
                    self._clients.discard(client)

    def close_all(self) -> None:
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            try:
                client.close()
            except OSError:
                continue


def run_embedding_once(text: str, config: BenchConfig) -> list[float] | None:
    backend = str(config.backend or "openvino").strip().lower() or "openvino"
    os.environ["EMBEDDINGS_BACKEND"] = backend

    model = str(config.model or "").strip() or None
    vector = _embed_text(text, model=model)
    if not isinstance(vector, list) or not vector:
        return None

    normalized = [float(value) for value in vector]
    if config.target_dim > 0:
        resized = _eta_mu_resize_vector(normalized, int(config.target_dim))
        normalized = _eta_mu_normalize_vector(resized)
    return normalized


class BenchmarkWorker(threading.Thread):
    def __init__(
        self,
        *,
        state: BenchmarkState,
        hub: WebSocketHub,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True, name="embed-bench-worker")
        self._state = state
        self._hub = hub
        self._stop_event = stop_event
        self._texts = cycle(BENCH_TEXTS)

    def run(self) -> None:
        while not self._stop_event.is_set():
            if not self._state.is_running():
                self._stop_event.wait(0.2)
                continue

            config = self._state.current_config()
            sample = next(self._texts)
            if config.max_chars > 0:
                sample = sample[: config.max_chars]

            started = time.perf_counter()
            ok = False
            vector_dim = 0
            error = ""

            try:
                vector = run_embedding_once(sample, config)
                latency_ms = (time.perf_counter() - started) * 1000.0
                if isinstance(vector, list) and vector:
                    ok = True
                    vector_dim = len(vector)
                else:
                    error = "embedding_none"
            except Exception as exc:  # pragma: no cover - defensive path
                latency_ms = (time.perf_counter() - started) * 1000.0
                error = f"{exc.__class__.__name__}:{exc}"[:220]

            event = self._state.record_sample(
                ok=ok,
                latency_ms=latency_ms,
                vector_dim=vector_dim,
                error=error,
                sample_text=sample,
            )
            self._hub.broadcast(
                {
                    "type": "sample",
                    "event": event,
                    "state": self._state.snapshot(include_recent=False),
                    "clients": self._hub.count(),
                }
            )

            interval_seconds = max(0.0, float(config.interval_ms) / 1000.0)
            if interval_seconds > 0:
                self._stop_event.wait(interval_seconds)


def websocket_accept_value(client_key: str) -> str:
    accept_seed = client_key + WS_MAGIC
    digest = hashlib.sha1(accept_seed.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("utf-8")


def websocket_frame(opcode: int, payload: bytes = b"") -> bytes:
    data = bytes(payload)
    length = len(data)
    header = bytearray([0x80 | (opcode & 0x0F)])
    if length <= 125:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + data


def websocket_frame_text(message: str) -> bytes:
    return websocket_frame(0x1, message.encode("utf-8"))


def _recv_ws_exact(connection: socket.socket, size: int) -> bytes | None:
    if size <= 0:
        return b""
    data = bytearray()
    while len(data) < size:
        try:
            chunk = connection.recv(size - len(data))
        except socket.timeout:
            if not data:
                raise
            continue
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def _read_ws_client_frame(connection: socket.socket) -> tuple[int, bytes] | None:
    header = _recv_ws_exact(connection, 2)
    if header is None:
        return None

    first, second = header
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    payload_len = second & 0x7F

    if payload_len == 126:
        extended = _recv_ws_exact(connection, 2)
        if extended is None:
            return None
        payload_len = struct.unpack("!H", extended)[0]
    elif payload_len == 127:
        extended = _recv_ws_exact(connection, 8)
        if extended is None:
            return None
        payload_len = struct.unpack("!Q", extended)[0]

    if not masked or payload_len > WS_CLIENT_FRAME_MAX_BYTES:
        return None

    mask_key = _recv_ws_exact(connection, 4)
    if mask_key is None:
        return None
    payload = _recv_ws_exact(connection, payload_len)
    if payload is None:
        return None

    if payload_len:
        payload = bytes(
            byte ^ mask_key[index % 4] for index, byte in enumerate(payload)
        )

    return opcode, payload


def _consume_ws_client_frame(connection: socket.socket) -> bool:
    frame = _read_ws_client_frame(connection)
    if frame is None:
        return False

    opcode, payload = frame
    if opcode == 0x8:
        try:
            connection.sendall(websocket_frame(0x8, payload[:125]))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        return False
    if opcode == 0x9:
        connection.sendall(websocket_frame(0xA, payload[:125]))
        return True
    if opcode in {0x0, 0x1, 0x2, 0xA}:
        return True
    return False


class EmbedBenchHandler(BaseHTTPRequestHandler):
    state: BenchmarkState
    hub: WebSocketHub
    shutdown_event: threading.Event

    protocol_version = "HTTP/1.1"

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        *,
        status: int = HTTPStatus.OK,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self._set_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if isinstance(extra_headers, dict):
            for key, value in extra_headers.items():
                self.send_header(str(key), str(value))
        self.end_headers()
        if body:
            try:
                self.wfile.write(body)
            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
                OSError,
            ):
                pass

    def _send_json(
        self, payload: dict[str, Any], *, status: int = HTTPStatus.OK
    ) -> None:
        self._send_bytes(
            _json_compact(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}

    def _send_ws_event(self, payload: dict[str, Any]) -> None:
        frame = websocket_frame_text(_json_compact(payload))
        self.connection.sendall(frame)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_bytes(b"", "text/plain; charset=utf-8", status=HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            self._send_bytes(
                HTML_INDEX.encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return

        if path == "/api/state":
            self._send_json(self.state.snapshot(include_recent=True))
            return

        if path == "/api/events":
            params = parse_qs(parsed.query)
            limit = _clamp_int(
                params.get("limit", ["50"])[0], default=50, min_value=1, max_value=200
            )
            snapshot = self.state.snapshot(include_recent=True)
            events = snapshot.get("recent_events", [])
            self._send_json(
                {
                    "ok": True,
                    "limit": limit,
                    "events": list(events)[-limit:] if isinstance(events, list) else [],
                    "state": snapshot,
                }
            )
            return

        if path == "/ws":
            self._handle_websocket()
            return

        self._send_json(
            {"ok": False, "error": "not_found", "path": path},
            status=HTTPStatus.NOT_FOUND,
        )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        payload = self._read_json_body()

        if path == "/api/control/start":
            backend = payload.get("backend")
            model = payload.get("model")
            interval_ms = payload.get("interval_ms")
            target_dim = payload.get("target_dim")
            window_size = payload.get("window_size")
            max_chars = payload.get("max_chars")
            reset = _as_bool(payload.get("reset"), default=True)

            self.state.update_config(
                backend=str(backend).strip().lower() if backend is not None else None,
                model=str(model).strip() if model is not None else None,
                interval_ms=(
                    _clamp_int(interval_ms, default=120, min_value=0, max_value=60_000)
                    if interval_ms is not None
                    else None
                ),
                target_dim=(
                    _clamp_int(target_dim, default=128, min_value=0, max_value=8192)
                    if target_dim is not None
                    else None
                ),
                window_size=(
                    _clamp_int(window_size, default=512, min_value=16, max_value=20_000)
                    if window_size is not None
                    else None
                ),
                max_chars=(
                    _clamp_int(max_chars, default=1600, min_value=64, max_value=64_000)
                    if max_chars is not None
                    else None
                ),
            )
            snapshot = self.state.start(reset=reset)
            self.hub.broadcast(
                {
                    "type": "state",
                    "state": snapshot,
                    "clients": self.hub.count(),
                }
            )
            self._send_json({"ok": True, "state": snapshot})
            return

        if path == "/api/control/stop":
            snapshot = self.state.stop()
            self.hub.broadcast(
                {
                    "type": "state",
                    "state": snapshot,
                    "clients": self.hub.count(),
                }
            )
            self._send_json({"ok": True, "state": snapshot})
            return

        if path == "/api/control/reset":
            snapshot = self.state.reset()
            self.hub.broadcast(
                {
                    "type": "state",
                    "state": snapshot,
                    "clients": self.hub.count(),
                }
            )
            self._send_json({"ok": True, "state": snapshot})
            return

        self._send_json(
            {"ok": False, "error": "not_found", "path": path},
            status=HTTPStatus.NOT_FOUND,
        )

    def _handle_websocket(self) -> None:
        ws_key = str(self.headers.get("Sec-WebSocket-Key", "")).strip()
        if not ws_key:
            self._send_bytes(
                b"missing websocket key",
                "text/plain; charset=utf-8",
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self._set_cors_headers()
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", websocket_accept_value(ws_key))
        self.end_headers()

        self.close_connection = True
        self.connection.settimeout(1.0)
        self.hub.add(self.connection)
        try:
            self._send_ws_event(
                {
                    "type": "snapshot",
                    "state": self.state.snapshot(include_recent=True),
                    "clients": self.hub.count(),
                }
            )
            while not self.shutdown_event.is_set():
                try:
                    if not _consume_ws_client_frame(self.connection):
                        break
                except socket.timeout:
                    continue
                except (
                    BrokenPipeError,
                    ConnectionResetError,
                    ConnectionAbortedError,
                    OSError,
                ):
                    break
        finally:
            self.hub.remove(self.connection)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long-running embedding benchmark stream server"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--backend",
        default=str(
            os.getenv(
                "EMBED_BENCH_BACKEND", os.getenv("EMBEDDINGS_BACKEND", "openvino")
            )
        ).strip()
        or "openvino",
    )
    parser.add_argument(
        "--model",
        default=str(
            os.getenv("EMBED_BENCH_MODEL", os.getenv("OPENVINO_EMBED_MODEL", ""))
        ).strip(),
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=_clamp_int(
            os.getenv("EMBED_BENCH_INTERVAL_MS", "120"),
            default=120,
            min_value=0,
            max_value=60_000,
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=_clamp_int(
            os.getenv("EMBED_BENCH_WINDOW_SIZE", "512"),
            default=512,
            min_value=16,
            max_value=20_000,
        ),
    )
    parser.add_argument(
        "--target-dim",
        type=int,
        default=_clamp_int(
            os.getenv("EMBED_BENCH_TARGET_DIM", "128"),
            default=128,
            min_value=0,
            max_value=8192,
        ),
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=_clamp_int(
            os.getenv("EMBED_BENCH_MAX_CHARS", "1600"),
            default=1600,
            min_value=64,
            max_value=64_000,
        ),
    )
    parser.add_argument(
        "--autostart",
        action="store_true",
        default=_as_bool(os.getenv("EMBED_BENCH_AUTOSTART", "1"), default=True),
        help="Start benchmark worker immediately",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = BenchConfig(
        backend=str(args.backend).strip().lower() or "openvino",
        model=str(args.model).strip(),
        interval_ms=_clamp_int(
            args.interval_ms, default=120, min_value=0, max_value=60_000
        ),
        window_size=_clamp_int(
            args.window_size, default=512, min_value=16, max_value=20_000
        ),
        target_dim=_clamp_int(
            args.target_dim, default=128, min_value=0, max_value=8192
        ),
        max_chars=_clamp_int(
            args.max_chars, default=1600, min_value=64, max_value=64_000
        ),
    )

    state = BenchmarkState(config)
    hub = WebSocketHub()
    shutdown_event = threading.Event()

    EmbedBenchHandler.state = state
    EmbedBenchHandler.hub = hub
    EmbedBenchHandler.shutdown_event = shutdown_event

    server = ThreadingHTTPServer((args.host, int(args.port)), EmbedBenchHandler)
    server.daemon_threads = True

    worker = BenchmarkWorker(state=state, hub=hub, stop_event=shutdown_event)
    worker.start()

    if bool(args.autostart):
        snapshot = state.start(reset=True)
        hub.broadcast(
            {
                "type": "state",
                "state": snapshot,
                "clients": hub.count(),
            }
        )

    print(
        "embedding bench server listening on "
        f"http://{args.host}:{args.port} "
        f"(backend={config.backend}, autostart={bool(args.autostart)})"
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        state.stop()
        try:
            server.shutdown()
        except OSError:
            pass
        server.server_close()
        hub.close_all()
        worker.join(timeout=3.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
