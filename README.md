# GeoShift

**Geographic domain-generalization, uncertainty calibration, and selective prediction for everyday object classification (`MLap/GeoDE`).**

GeoShift measures how reliably pretrained vision models generalize when evaluated on a **completely unseen geographic region**—and provides a containerized **FastAPI + Pydantic** inference service that abstains on low-confidence out-of-distribution (OOD) inputs instead of guessing wrong.

---

## 1. Problem & Experimental Design

Standard random train/test splits hide geographic bias. GeoShift evaluates **leave-one-region-out** generalization across the **GeoDE** dataset ([`MLap/GeoDE`](https://huggingface.co/datasets/MLap/GeoDE), pinned revision `acbae31823b22f84c3bbb11585d37d50aabaac44`):

- **Dataset:** `61,925` everyday object photographs in the Hugging Face mirror across **40 object classes**, **6 geographic regions**, and **43 raw countries** *(note: the original GeoDE paper reports ~61,940 images; this project uses and documents the verified 61,925-sample mirror)*.
- **6 Geographic Holdouts:** For each target region (`Africa`, `Americas`, `EastAsia`, `Europe`, `SouthEastAsia`, `WestAsia`), all samples from that region are strictly held out for OOD testing, while the remaining 5 regions are split `85% / 15%` (stratified by `label × region`, `seed=42`) into training and in-distribution validation.
- **Models Compared:** Parameter-matched ImageNet-pretrained backbones trained with shuffled mini-batch linear probes:
  - **CNN:** `resnet50` (~25.6M params, 2048-d features)
  - **Vision Transformer:** `swin_t` (~28.3M params, 768-d features)

---

## 2. Key Findings (6-Fold Leave-One-Region-Out)

### A. ResNet-50 (CNN) vs. Swin-Tiny (Vision Transformer)

| Held-Out Region | ResNet-50 In-Dist | ResNet-50 OOD | ResNet-50 Gap | Swin-T In-Dist | Swin-T OOD | Swin-T Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Africa** | 91.97% | **87.81%** | +4.16 pp | 91.86% | 87.68% | +4.18 pp |
| **EastAsia** | 91.31% | 88.91% | +2.40 pp | 91.35% | **89.14%** | +2.22 pp |
| **SouthEastAsia** | 90.65% | 89.66% | +0.99 pp | 91.12% | **89.86%** | +1.26 pp |
| **Americas** | 91.57% | **91.21%** | +0.36 pp | 91.47% | 91.00% | +0.47 pp |
| **WestAsia** | 91.00% | 91.03% | -0.03 pp | 91.64% | **91.38%** | +0.26 pp |
| **Europe** | 91.28% | 91.59% | -0.31 pp | 91.09% | **91.92%** | -0.83 pp |
| **6-Fold Mean** | **91.30%** | **90.03%** | **+1.26 pp** | **91.42%** | **90.16%** | **+1.26 pp** |

- **Asymmetric geographic shift:** Both CNN and ViT models generalize well to `Europe` (even slightly outperforming rest-of-world validation) while suffering their steepest drops on `Africa` (`~4.2 pp`) and `EastAsia` (`~2.2–2.4 pp`).

### B. Per-Class Failure Analysis (Held-Out: Africa)

Aggregate accuracy hides severe class-specific shift on culturally and regionally distinct objects:

| Object Class | ResNet-50 In-Dist → OOD (Drop) | Swin-T In-Dist → OOD (Drop) |
| :--- | :---: | :---: |
| `religious_building` | 87.65% → 64.35% (**-23.31 pp**) | 91.98% → 54.35% (**-37.63 pp**) |
| `bus` | 96.91% → 76.25% (**-20.66 pp**) | 98.15% → 72.50% (**-25.65 pp**) |
| `fence` | 88.32% → 65.84% (**-22.49 pp**) | 88.32% → 66.77% (**-21.55 pp**) |
| `storefront` | 87.73% → 76.14% (**-11.59 pp**) | 87.12% → 74.51% (**-12.61 pp**) |

Although `swin_t` edges out `resnet50` in mean OOD accuracy (`90.16%` vs `90.03%`), `resnet50` exhibits lower OOD Expected Calibration Error (`0.0088` vs `0.0108`) and less severe per-class collapse on `religious_building` and `bus` in Africa.

### C. Calibration & Selective Prediction

- **In-distribution temperature scaling does not automatically transfer OOD:** Because frozen-backbone linear probes are already well-calibrated in-distribution, LBFGS temperature scaling yields `T < 1` (`0.914–0.965` for ResNet-50; `0.926–1.006` for Swin-T). Sharpening logits improves `Europe`/`Americas`, **but worsens Africa OOD ECE** (`0.0081 → 0.0183` for ResNet-50; `0.0196 → 0.0225` for Swin-T).
- **Selective prediction reliably recovers OOD accuracy:** Abstaining on low-confidence predictions boosts OOD accuracy across all 6 regions:
  - **ResNet-50 (6-fold mean):** `90.03%` (100% cov) → **`94.41%`** (90% cov) → **`96.98%`** (80% cov)
  - **Swin-T (6-fold mean):** `90.16%` (100% cov) → **`94.54%`** (90% cov) → **`97.09%`** (80% cov)
  - **Africa (ResNet-50):** `87.81%` (100% cov) → **`92.41%`** (90% cov) → **`95.31%`** (80% cov)

The analysis script generates `outputs/figures/regional_generalization_gap.png` and `outputs/figures/selective_prediction_curves.png`. Generated output figures are not included in this repository.

---

## 3. Repository Structure

```text
geoshift/
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── scripts/
│   ├── run_data_pipeline.py        # Stream metadata, clean, validate, profile, build 6 manifests
│   ├── extract_features.py         # Single-pass O(1)-chunked GPU feature extractor
│   ├── run_linear_probe.py         # 6-fold shuffled linear probe training & evaluation
│   ├── run_reliability_eval.py     # Temperature scaling, ECE/Brier/NLL, accuracy-vs-coverage
│   ├── analyze_results.py          # Per-class drop analysis & matplotlib charts
│   ├── train_baseline.py           # End-to-end streaming trainer (frozen or fine-tuned)
│   ├── benchmark_data_pipeline.py  # Streaming throughput benchmark
│   └── smoke_test_geode.py         # Live Hugging Face smoke test
├── src/geoshift/
│   ├── data/                       # GeoDE stream adapter, ingestion, cleaning, validation, splits, dataset
│   ├── models/                     # Unified GeoClassifier (resnet50, convnext_tiny, efficientnet_b0, swin_t)
│   ├── training/                   # AMP training/validation loop & atomic checkpointing
│   ├── evaluation/                 # Classification metrics, ECE, Brier, NLL, selective prediction
│   ├── inference/                  # Calibrated GeoShiftPredictor with coverage-based abstention
│   └── api/                        # FastAPI + Pydantic service, web UI (/), and /metrics monitor
└── tests/                          # Unit & integration tests across data, models, training, eval, inference, api
```

## 4. Quickstart (Local)

Activate the project environment and run the test suite:

```powershell
conda activate geoshift
python -m pytest -v
```

### Configuration

Runtime and experiment defaults are centralized in [`configs/`](configs/):

- `data.yml` — GeoDE revision, geographic regions, split labels/seed, cleaning aliases, and image transforms
- `models.yml` — default and supported backbones
- `training.yml` — feature extraction, linear-probe, and baseline training settings
- `evaluation.yml` — calibration, selective-prediction, analysis, and benchmark settings
- `inference.yml` — API, predictor, monitoring, and coverage-mode settings
- `paths.yml` — data and output locations

Edit these YAML files to change defaults. CLI options override experiment settings; `GEOSHIFT_CONFIG_DIR` selects a different configuration directory for deployments. The API also supports `GEOSHIFT_FOLD_DIR`, `GEOSHIFT_BACKBONE`, `GEOSHIFT_TARGET_COVERAGE`, `GEOSHIFT_LOG_PATH`, `GEOSHIFT_API_HOST`, and `GEOSHIFT_API_PORT` environment overrides.

### Reproduce the Pipeline & Experiments

Run these commands from the repository root in PowerShell:

```powershell
# 1. Ingest metadata & build the 6 geographic holdout manifests
python scripts\run_data_pipeline.py

# 2. Extract frozen-backbone features in a single resumable pass
python scripts\extract_features.py --backbone resnet50

# 3. Train & evaluate across all 6 geographic folds
python scripts\run_linear_probe.py

# 4. Run calibration & selective prediction
python scripts\run_reliability_eval.py

# 5. Generate per-class error analysis & figures
python scripts\analyze_results.py
```

### Run the Inference API & Web UI

```powershell
python -m geoshift.api
```

Open `http://127.0.0.1:8000` for the interactive classifier UI (with selectable `80% / 90% / 100%` reliability coverage modes and live telemetry), `/docs` for OpenAPI/Swagger, `/health` for readiness checks, and `/metrics` for live abstention-rate and p95 latency monitoring.

## 5. Containerization & AWS Deployment (ECR + App Runner)

Build and run the CPU inference container locally:

```powershell
docker build -t geoshift:latest .
docker run -p 8000:8000 geoshift:latest
```

Push the image to Amazon ECR:

```powershell
aws ecr create-repository --repository-name geoshift --region ap-south-1
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com
docker tag geoshift:latest <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/geoshift:latest
docker push <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/geoshift:latest
```

Then create an **App Runner** service pointing to `<ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/geoshift:latest` with port `8000` and HTTP health check `/health`.

## 6. Limitations & Next Steps

- **Frozen-backbone scope:** The 6-fold results above evaluate frozen ImageNet representations (`resnet50` and `swin_t`); end-to-end backbone fine-tuning (`scripts/train_baseline.py --unfreeze`) on `Africa` remains a natural next experiment.
- **Dataset licensing:** Verify `MLap/GeoDE` citation and licensing terms before any commercial redistribution.
- **Single-worker streaming:** `GeoDEIterableDataset` enforces `num_workers=0` when streaming directly from Hugging Face.
