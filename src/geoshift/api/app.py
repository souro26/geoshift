"""FastAPI inference service with Pydantic schemas and a lightweight upload UI."""

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from geoshift.inference import GeoShiftPredictor


DEFAULT_FOLD_DIR = Path("outputs/experiments/linear_probe_resnet50/heldout_Africa")


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
    top_predictions: list[TopPrediction]


_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GeoShift — Geographic OOD Classifier</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 680px; margin: 2.5rem auto; padding: 0 1rem; color: #111827; background: #f9fafb; }
    .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem; }
    h1 { margin: 0 0 0.25rem; font-size: 1.5rem; }
    .sub { color: #4b5563; margin: 0 0 1rem; font-size: 0.95rem; }
    button { background: #111827; color: #fff; border: none; padding: 0.6rem 1.2rem; border-radius: 6px; cursor: pointer; font-weight: 600; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
    .ok { background: #d1fae5; color: #065f46; }
    .warn { background: #fef3c7; color: #92400e; }
    .bar-row { display: flex; align-items: center; gap: 0.75rem; margin: 0.4rem 0; font-size: 0.9rem; }
    .bar-label { width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .bar-track { flex: 1; background: #e5e7eb; height: 10px; border-radius: 999px; overflow: hidden; }
    .bar-fill { background: #2563eb; height: 100%; }
    img#preview { max-width: 100%; max-height: 240px; border-radius: 8px; margin-top: 1rem; display: none; }
  </style>
</head>
<body>
  <div class="card">
    <h1>GeoShift</h1>
    <p class="sub">Calibrated GeoDE object classifier with selective-prediction abstention.</p>
    <input id="fileInput" type="file" accept="image/*" />
    <button id="predictBtn" disabled>Predict</button>
    <img id="preview" alt="Uploaded preview" />
  </div>
  <div id="resultCard" class="card" style="display:none;"></div>
  <script>
    const fileInput = document.getElementById("fileInput");
    const predictBtn = document.getElementById("predictBtn");
    const preview = document.getElementById("preview");
    const resultCard = document.getElementById("resultCard");

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
      const formData = new FormData();
      formData.append("file", file);
      try {
        const res = await fetch("/predict?top_k=5", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Prediction failed");
        const badge = data.abstain
          ? `<span class="badge warn">ABSTAIN (Low Confidence &lt; ${(data.confidence_threshold * 100).toFixed(1)}%)</span>`
          : `<span class="badge ok">ACCEPTED</span>`;
        const bars = data.top_predictions.map(p => `
          <div class="bar-row">
            <span class="bar-label">${p.label}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${(p.probability * 100).toFixed(1)}%"></div></div>
            <span>${(p.probability * 100).toFixed(1)}%</span>
          </div>`).join("");
        resultCard.innerHTML = `
          <div><strong>Prediction:</strong> ${data.predicted_class} (${(data.confidence * 100).toFixed(2)}%) ${badge}</div>
          <p class="sub" style="margin-top:0.4rem;">Temperature: ${data.temperature.toFixed(3)}</p>
          ${bars}`;
        resultCard.style.display = "block";
      } catch (err) {
        resultCard.innerHTML = `<strong>Error:</strong> ${err.message}`;
        resultCard.style.display = "block";
      } finally {
        predictBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def create_app(
    predictor: GeoShiftPredictor | None = None,
    fold_dir: str | Path | None = None,
    backbone_name: str | None = None,
    target_coverage: float = 0.90,
) -> FastAPI:
    """Create a FastAPI app wrapping a GeoShiftPredictor instance."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.predictor is None:
            resolved_dir = Path(
                fold_dir or os.environ.get("GEOSHIFT_FOLD_DIR", str(DEFAULT_FOLD_DIR))
            )
            resolved_backbone = backbone_name or os.environ.get(
                "GEOSHIFT_BACKBONE", "resnet50"
            )
            application.state.predictor = GeoShiftPredictor.from_fold_dir(
                fold_dir=resolved_dir,
                backbone_name=resolved_backbone,
                target_coverage=target_coverage,
            )
        yield

    application = FastAPI(
        title="GeoShift Inference API",
        version="0.1.0",
        description="Geographic domain-generalization image classifier with calibration and abstention.",
        lifespan=lifespan,
    )
    application.state.predictor = predictor

    @application.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

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

    @application.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: UploadFile = File(...),
        top_k: int = Query(default=5, ge=1, le=40),
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

        result = active_predictor.predict_image(image, top_k=top_k)
        return PredictionResponse(**result)

    return application


app = create_app()