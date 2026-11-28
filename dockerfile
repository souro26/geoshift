FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GEOSHIFT_BACKBONE=resnet50 \
    GEOSHIFT_FOLD_DIR=/app/outputs/experiments/linear_probe_resnet50/heldout_Africa

WORKDIR /app

# Install CPU-only PyTorch + torchvision for lean cloud serving
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# Install API and runtime dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pillow \
    numpy \
    pandas \
    pydantic

# Pre-cache both Champion (resnet50) and Challenger (swin_t) weights
RUN python -c "from torchvision import models; models.resnet50(weights=models.ResNet50_Weights.DEFAULT); models.swin_t(weights=models.Swin_T_Weights.DEFAULT)"

COPY pyproject.toml ./
COPY src ./src

# Bundle lightweight linear-probe heads + calibration configs (~1 MB total)
COPY outputs/experiments/linear_probe_resnet50/heldout_Africa/logits_and_head.pt \
     outputs/experiments/linear_probe_resnet50/heldout_Africa/reliability.json \
     ./outputs/experiments/linear_probe_resnet50/heldout_Africa/
COPY outputs/experiments/linear_probe_swin_t/heldout_Africa/logits_and_head.pt \
     outputs/experiments/linear_probe_swin_t/heldout_Africa/reliability.json \
     ./outputs/experiments/linear_probe_swin_t/heldout_Africa/

RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "geoshift.api.app:app", "--host", "0.0.0.0", "--port", "8000"]