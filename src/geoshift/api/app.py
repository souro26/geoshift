"""FastAPI inference service with Pydantic schemas, monitoring, and user UI."""

import io
import json
import os
import statistics
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from geoshift.config import load_config, project_path
from geoshift.inference import GeoShiftPredictor

_INFERENCE_CONFIG = load_config("inference.yml")
_MODEL_CONFIG = load_config("models.yml")
DEFAULT_FOLD_DIR = project_path(_INFERENCE_CONFIG["fold_dir"])
DEFAULT_LOG_PATH = project_path(_INFERENCE_CONFIG["log_path"])


class HealthResponse(BaseModel):
    """Service health status."""

    status: str = Field(..., examples=["ok"])
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Active model, calibration, and selective-prediction metadata."""

    backbone: str
    heldout_region: str | None
    num_classes: int
    temperature: float
    target_coverage: float | None
    confidence_threshold: float | None
    device: str


class TopPrediction(BaseModel):
    """Single ranked class prediction."""

    label: str
    probability: float


class PredictionResponse(BaseModel):
    """Calibrated classification response with abstention signal."""

    predicted_class: str
    confidence: float
    abstain: bool
    confidence_threshold: float | None
    temperature: float
    latency_ms: float
    top_predictions: list[TopPrediction]


class MetricsResponse(BaseModel):
    """Operational telemetry and drift-proxy monitoring metrics."""

    total_requests: int
    accepted_count: int
    abstained_count: int
    abstention_rate: float
    avg_confidence: float
    avg_latency_ms: float
    p95_latency_ms: float
    top_predicted_classes: dict[str, int]


class PredictionMonitor:
    """Thread-safe prediction telemetry tracker and JSONL logger."""

    def __init__(
        self,
        log_path: Path | None = None,
        window_size: int = _INFERENCE_CONFIG["monitor_window_size"],
    ) -> None:
        self._lock = Lock()
        self._log_path = log_path
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._confidences: deque[float] = deque(maxlen=window_size)
        self._class_counts: Counter[str] = Counter()
        self._total_requests = 0
        self._abstained_count = 0

    def record(
        self,
        predicted_class: str,
        confidence: float,
        abstain: bool,
        latency_ms: float,
    ) -> None:
        with self._lock:
            self._total_requests += 1
            if abstain:
                self._abstained_count += 1
            self._latencies.append(latency_ms)
            self._confidences.append(confidence)
            self._class_counts[predicted_class] += 1

            if self._log_path is not None:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                event = {
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "predicted_class": predicted_class,
                    "confidence": round(confidence, 6),
                    "abstain": abstain,
                    "latency_ms": round(latency_ms, 3),
                }
                with self._log_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(event) + "\n")

    def snapshot(self) -> MetricsResponse:
        with self._lock:
            total = self._total_requests
            abstained = self._abstained_count
            accepted = total - abstained
            abstention_rate = round(abstained / total, 6) if total > 0 else 0.0
            avg_conf = (
                round(statistics.fmean(self._confidences), 6)
                if self._confidences
                else 0.0
            )
            avg_lat = (
                round(statistics.fmean(self._latencies), 3)
                if self._latencies
                else 0.0
            )
            if self._latencies:
                sorted_lats = sorted(self._latencies)
                p95_idx = min(len(sorted_lats) - 1, int(0.95 * (len(sorted_lats) - 1)))
                p95_lat = round(sorted_lats[p95_idx], 3)
            else:
                p95_lat = 0.0

            return MetricsResponse(
                total_requests=total,
                accepted_count=accepted,
                abstained_count=abstained,
                abstention_rate=abstention_rate,
                avg_confidence=avg_conf,
                avg_latency_ms=avg_lat,
                p95_latency_ms=p95_lat,
                top_predicted_classes=dict(
                    self._class_counts.most_common(
                        _INFERENCE_CONFIG["top_predicted_classes"]
                    )
                ),
            )


def _load_reliability_curve(fold_dir: Path) -> list[dict[str, Any]]:
    reliability_path = fold_dir / "reliability.json"
    if not reliability_path.exists():
        return []
    payload = json.loads(reliability_path.read_text(encoding="utf-8"))
    return list(payload.get("ood_selective_prediction", []))


_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GeoShift — Geographic Shift-Aware Classifier</title>
  <style>
    :root { --bg: #f8fafc; --card: #ffffff; --border: #e2e8f0; --text: #0f172a; --muted: #475569; --accent: #2563eb; }
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 820px; margin: 2rem auto; padding: 0 1rem; color: var(--text); background: var(--bg); }
    .header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
    h1 { margin: 0; font-size: 1.6rem; }
    .sub { color: var(--muted); margin: 0.25rem 0 0; font-size: 0.95rem; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; }
    .card h2 { margin: 0 0 0.75rem; font-size: 1.05rem; }
    button { background: var(--text); color: #fff; border: none; padding: 0.6rem 1.2rem; border-radius: 8px; cursor: pointer; font-weight: 600; }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    .control-row { margin: 0.9rem 0; font-size: 0.9rem; color: var(--muted); }
    select, input[type="file"] { margin-top: 0.35rem; display: block; width: 100%; }
    .badge { display: inline-block; padding: 0.25rem 0.65rem; border-radius: 999px; font-size: 0.78rem; font-weight: 700; }
    .ok { background: #dcfce7; color: #166534; }
    .warn { background: #fef3c7; color: #92400e; }
    .bar-row { display: flex; align-items: center; gap: 0.65rem; margin: 0.45rem 0; font-size: 0.88rem; }
    .bar-label { width: 135px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .bar-track { flex: 1; background: #e2e8f0; height: 9px; border-radius: 999px; overflow: hidden; }
    .bar-fill { background: var(--accent); height: 100%; }
    img#preview { max-width: 100%; max-height: 220px; border-radius: 8px; margin-top: 0.9rem; display: none; border: 1px solid var(--border); }
    .kpi-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.65rem; }
    .kpi { background: #f1f5f9; border-radius: 8px; padding: 0.65rem; }
    .kpi .val { font-size: 1.15rem; font-weight: 700; }
    .kpi .lbl { font-size: 0.78rem; color: var(--muted); }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>GeoShift</h1>
      <p class="sub">Everyday object recognition with calibrated confidence &amp; selective abstention</p>
    </div>
    <a href="/docs" style="font-size:0.88rem; color:var(--accent); text-decoration:none;">API Docs (/docs)</a>
  </div>

  <div class="grid">
    <div class="card">
      <h2>1. Upload Image</h2>
      <input id="fileInput" type="file" accept="image/*" />
      <div class="control-row">
        <label for="coverageSelect"><strong>Reliability Mode (Selective Prediction)</strong></label>
        <select id="coverageSelect">
            {{COVERAGE_OPTIONS}}
          </select>
      </div>
      <button id="predictBtn" disabled>Classify Object</button>
      <img id="preview" alt="Uploaded preview" />
    </div>

    <div class="card">
      <h2>Live Service Telemetry</h2>
      <div id="metricsBox" class="kpi-grid">
        <div class="kpi"><div class="val" id="mTotal">-</div><div class="lbl">Total Requests</div></div>
        <div class="kpi"><div class="val" id="mAbstain">-</div><div class="lbl">Abstention Rate</div></div>
        <div class="kpi"><div class="val" id="mConf">-</div><div class="lbl">Avg Confidence</div></div>
        <div class="kpi"><div class="val" id="mLat">-</div><div class="lbl">p95 Latency (ms)</div></div>
      </div>
      <p class="sub" id="modelMeta" style="margin-top:0.85rem; font-size:0.82rem;"></p>
    </div>
  </div>

  <div id="resultCard" class="card" style="display:none;"></div>

  <script>
    const fileInput = document.getElementById("fileInput");
    const predictBtn = document.getElementById("predictBtn");
    const coverageSelect = document.getElementById("coverageSelect");
    const preview = document.getElementById("preview");
    const resultCard = document.getElementById("resultCard");
    let reliabilityCurve = [];

    async function refreshTelemetry() {
      try {
        const [mRes, infoRes, relRes] = await Promise.all([
          fetch("/metrics"),
          fetch("/model-info"),
          fetch("/reliability")
        ]);
        if (mRes.ok) {
          const m = await mRes.json();
          document.getElementById("mTotal").textContent = m.total_requests;
          document.getElementById("mAbstain").textContent = (m.abstention_rate * 100).toFixed(1) + "%";
          document.getElementById("mConf").textContent = (m.avg_confidence * 100).toFixed(1) + "%";
          document.getElementById("mLat").textContent = m.p95_latency_ms.toFixed(1);
        }
        if (infoRes.ok) {
          const info = await infoRes.json();
          document.getElementById("modelMeta").textContent =
            `Backbone: ${info.backbone} | Held-out Fold: ${info.heldout_region || "custom"} | Classes: ${info.num_classes} | T=${info.temperature.toFixed(3)}`;
        }
        if (relRes.ok) {
          const rel = await relRes.json();
          reliabilityCurve = rel.curve || [];
        }
      } catch (_) {}
    }

    function thresholdForCoverage(cov) {
      if (cov >= 1.0) return 0.0;
      const match = reliabilityCurve.find(p => Math.abs(p.target_coverage - cov) < 1e-4);
      return match ? match.confidence_threshold : null;
    }

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      predictBtn.disabled = !file;
      if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
      }
    });

    predictBtn.addEventListener("click", async () => {
      const file = fileInput.files[0];
      if (!file) return;
      predictBtn.disabled = true;
      const cov = parseFloat(coverageSelect.value);
      const threshold = thresholdForCoverage(cov);
      const query = threshold !== null
        ? `?top_k={{TOP_K}}&confidence_threshold=${threshold}`
        : "?top_k={{TOP_K}}";
      const formData = new FormData();
      formData.append("file", file);
      try {
        const res = await fetch(`/predict${query}`, { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Prediction failed");
        const badge = data.abstain
          ? `<span class="badge warn">ABSTAIN — Low Confidence (&lt; ${(data.confidence_threshold * 100).toFixed(1)}%)</span>`
          : `<span class="badge ok">ACCEPTED</span>`;
        const bars = data.top_predictions.map(p => `
          <div class="bar-row">
            <span class="bar-label">${p.label}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${(p.probability * 100).toFixed(1)}%"></div></div>
            <span>${(p.probability * 100).toFixed(1)}%</span>
          </div>`).join("");
        resultCard.innerHTML = `
          <h2>Prediction Result</h2>
          <div><strong>${data.predicted_class}</strong> (${(data.confidence * 100).toFixed(2)}%) ${badge}</div>
          <p class="sub" style="margin-top:0.35rem; font-size:0.84rem;">Latency: ${data.latency_ms.toFixed(1)} ms | Temperature: ${data.temperature.toFixed(3)}</p>
          ${bars}`;
        resultCard.style.display = "block";
        refreshTelemetry();
      } catch (err) {
        resultCard.innerHTML = `<strong>Error:</strong> ${err.message}`;
        resultCard.style.display = "block";
      } finally {
        predictBtn.disabled = false;
      }
    });

    refreshTelemetry();
  </script>
</body>
</html>
"""


def _render_index_html() -> str:
    """Render coverage choices from the active service configuration."""

    options = "\n".join(
        (
            f'<option value="{mode["coverage"]:.2f}">'
            f'{mode["label"]} ({mode["description"]})</option>'
        )
        for mode in _INFERENCE_CONFIG["coverage_modes"]
    )
    return (
        _INDEX_HTML.replace("{{COVERAGE_OPTIONS}}", options)
        .replace("{{TOP_K}}", str(_INFERENCE_CONFIG["top_k"]))
    )


def create_app(
    predictor: GeoShiftPredictor | None = None,
    fold_dir: str | Path | None = None,
    backbone_name: str | None = None,
    target_coverage: float = _INFERENCE_CONFIG["target_coverage"],
    log_path: str | Path | None = DEFAULT_LOG_PATH,
) -> FastAPI:
    """Create a FastAPI app wrapping a GeoShiftPredictor and live monitor."""

    resolved_fold_dir = Path(
        fold_dir or os.environ.get("GEOSHIFT_FOLD_DIR", str(DEFAULT_FOLD_DIR))
    )
    resolved_backbone = backbone_name or os.environ.get(
        "GEOSHIFT_BACKBONE", _MODEL_CONFIG["default_backbone"]
    )
    resolved_target_coverage = float(
        os.environ.get("GEOSHIFT_TARGET_COVERAGE", target_coverage)
    )
    environment_log_path = os.environ.get("GEOSHIFT_LOG_PATH")
    if log_path == DEFAULT_LOG_PATH and environment_log_path:
        resolved_log_path = Path(environment_log_path)
    else:
        resolved_log_path = Path(log_path) if log_path is not None else None

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.predictor is None:
            application.state.predictor = GeoShiftPredictor.from_fold_dir(
                fold_dir=resolved_fold_dir,
                backbone_name=resolved_backbone,
                target_coverage=resolved_target_coverage,
            )
            application.state.reliability_curve = _load_reliability_curve(
                resolved_fold_dir
            )
        yield

    application = FastAPI(
        title=_INFERENCE_CONFIG["api"]["title"],
        version=_INFERENCE_CONFIG["api"]["version"],
        description=_INFERENCE_CONFIG["api"]["description"],
        lifespan=lifespan,
    )
    application.state.predictor = predictor
    application.state.monitor = PredictionMonitor(log_path=resolved_log_path)
    application.state.reliability_curve = (
        _load_reliability_curve(resolved_fold_dir) if predictor is None else []
    )

    @application.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _render_index_html()

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=application.state.predictor is not None,
        )

    @application.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        active_predictor = application.state.predictor
        if active_predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        return ModelInfoResponse(**active_predictor.get_model_info())

    @application.get("/reliability")
    def reliability() -> dict[str, Any]:
        return {"curve": application.state.reliability_curve}

    @application.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        return application.state.monitor.snapshot()

    @application.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: UploadFile = File(...),
        top_k: int = Query(
            default=_INFERENCE_CONFIG["top_k"],
            ge=1,
            le=_INFERENCE_CONFIG["max_top_k"],
        ),
        confidence_threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    ) -> PredictionResponse:
        active_predictor = application.state.predictor
        if active_predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")

        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            image = Image.open(io.BytesIO(raw_bytes))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image.",
            ) from exc

        start = time.perf_counter()
        result = active_predictor.predict_image(image, top_k=top_k)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)

        effective_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else result["confidence_threshold"]
        )
        abstain = (
            result["confidence"] < effective_threshold
            if effective_threshold is not None
            else False
        )

        application.state.monitor.record(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            abstain=abstain,
            latency_ms=latency_ms,
        )

        return PredictionResponse(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            abstain=abstain,
            confidence_threshold=effective_threshold,
            temperature=result["temperature"],
            latency_ms=latency_ms,
            top_predictions=result["top_predictions"],
        )

    return application


app = create_app()


def run() -> None:
    """Run the service using its configured host and port."""

    import uvicorn

    api_config = _INFERENCE_CONFIG["api"]
    uvicorn.run(
        app,
        host=os.environ.get("GEOSHIFT_API_HOST", api_config["host"]),
        port=int(os.environ.get("GEOSHIFT_API_PORT", api_config["port"])),
    )


if __name__ == "__main__":
    run()