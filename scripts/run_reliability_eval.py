"""Evaluate calibration (temperature scaling) and selective prediction across folds."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from geoshift.config import load_config, project_path
from geoshift.evaluation import (
    DEFAULT_COVERAGES,
    apply_temperature,
    compute_brier_score,
    compute_expected_calibration_error,
    compute_negative_log_likelihood,
    compute_selective_prediction_curve,
    fit_temperature,
)

_DATA_CONFIG = load_config("data.yml")
_PATH_CONFIG = load_config("paths.yml")
_EVALUATION_CONFIG = load_config("evaluation.yml")
_ANALYSIS_CONFIG = _EVALUATION_CONFIG["analysis"]
ALL_REGIONS = tuple(_DATA_CONFIG["dataset"]["regions"])
DEFAULT_EXPERIMENT_DIR = project_path(_PATH_CONFIG["outputs"]["experiments"]) / (
    f"linear_probe_{_ANALYSIS_CONFIG['experiment_backbone']}"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for reliability evaluation."""

    parser = argparse.ArgumentParser(
        description="Run temperature scaling and selective prediction on saved fold logits."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Experiment directory containing heldout_<Region>/logits_and_head.pt.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=_EVALUATION_CONFIG["calibration"]["n_bins"],
        help="Number of confidence bins for ECE.",
    )
    return parser.parse_args()


def _calibration_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int,
) -> dict[str, float]:
    probs = F.softmax(logits.float(), dim=1).cpu().numpy()
    y_true = labels.cpu().numpy()
    return {
        "ece": round(compute_expected_calibration_error(probs, y_true, n_bins=n_bins), 6),
        "brier": round(compute_brier_score(probs, y_true), 6),
        "nll": round(compute_negative_log_likelihood(logits, labels), 6),
    }


def main() -> None:
    args = parse_args()

    if not args.experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {args.experiment_dir}")

    print("GeoShift | Calibration & Selective Prediction (Phase 4)")
    print(f"Experiment: {args.experiment_dir}\n")

    fold_summaries: list[dict] = []

    for region in ALL_REGIONS:
        fold_dir = args.experiment_dir / f"heldout_{region}"
        logits_path = fold_dir / "logits_and_head.pt"
        if not logits_path.exists():
            raise FileNotFoundError(f"Missing saved logits: {logits_path}")

        payload = torch.load(logits_path, map_location="cpu", weights_only=True)
        val_logits = payload["val_logits"].float()
        val_labels = payload["val_labels"].long()
        test_logits = payload["test_logits"].float()
        test_labels = payload["test_labels"].long()

        temperature = fit_temperature(val_logits, val_labels)

        val_before = _calibration_metrics(val_logits, val_labels, n_bins=args.n_bins)
        val_after = _calibration_metrics(
            apply_temperature(val_logits, temperature),
            val_labels,
            n_bins=args.n_bins,
        )

        ood_before = _calibration_metrics(test_logits, test_labels, n_bins=args.n_bins)
        scaled_test_logits = apply_temperature(test_logits, temperature)
        ood_after = _calibration_metrics(scaled_test_logits, test_labels, n_bins=args.n_bins)

        scaled_test_probs = F.softmax(scaled_test_logits, dim=1).cpu().numpy()
        ood_curve = compute_selective_prediction_curve(
            scaled_test_probs,
            test_labels.cpu().numpy(),
            coverages=DEFAULT_COVERAGES,
        )

        acc_by_cov = {
            f"{int(round(point['target_coverage'] * 100))}%": point["accuracy"]
            for point in ood_curve
        }

        fold_report = {
            "heldout_region": region,
            "temperature": round(temperature, 6),
            "in_dist_val_before": val_before,
            "in_dist_val_after": val_after,
            "ood_test_before": ood_before,
            "ood_test_after": ood_after,
            "ood_selective_prediction": ood_curve,
        }

        with (fold_dir / "reliability.json").open("w", encoding="utf-8") as file:
            json.dump(fold_report, file, indent=2)

        fold_summaries.append(
            {
                "region": region,
                "temperature": round(temperature, 4),
                "ood_ece_before": ood_before["ece"],
                "ood_ece_after": ood_after["ece"],
                "ood_nll_before": ood_before["nll"],
                "ood_nll_after": ood_after["nll"],
                "ood_brier_before": ood_before["brier"],
                "ood_brier_after": ood_after["brier"],
                "acc_at_100": acc_by_cov["100%"],
                "acc_at_90": acc_by_cov["90%"],
                "acc_at_80": acc_by_cov["80%"],
            }
        )

        print(
            f"{region:14} | T={temperature:.3f} | "
            f"OOD ECE: {ood_before['ece']:.4f} -> {ood_after['ece']:.4f} | "
            f"OOD Acc @100%/90%/80%: "
            f"{acc_by_cov['100%']:.4f} / {acc_by_cov['90%']:.4f} / {acc_by_cov['80%']:.4f}"
        )

    summary = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "experiment_dir": str(args.experiment_dir),
        "n_bins": args.n_bins,
        "mean_ood_ece_before": round(
            float(np.mean([f["ood_ece_before"] for f in fold_summaries])), 6
        ),
        "mean_ood_ece_after": round(
            float(np.mean([f["ood_ece_after"] for f in fold_summaries])), 6
        ),
        "mean_ood_nll_before": round(
            float(np.mean([f["ood_nll_before"] for f in fold_summaries])), 6
        ),
        "mean_ood_nll_after": round(
            float(np.mean([f["ood_nll_after"] for f in fold_summaries])), 6
        ),
        "mean_acc_at_100": round(
            float(np.mean([f["acc_at_100"] for f in fold_summaries])), 6
        ),
        "mean_acc_at_90": round(
            float(np.mean([f["acc_at_90"] for f in fold_summaries])), 6
        ),
        "mean_acc_at_80": round(
            float(np.mean([f["acc_at_80"] for f in fold_summaries])), 6
        ),
        "folds": fold_summaries,
    }

    summary_path = args.experiment_dir / "reliability_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nRELIABILITY SUMMARY")
    print(
        f"Mean OOD ECE:  {summary['mean_ood_ece_before']:.4f} -> "
        f"{summary['mean_ood_ece_after']:.4f}"
    )
    print(
        f"Mean OOD NLL:  {summary['mean_ood_nll_before']:.4f} -> "
        f"{summary['mean_ood_nll_after']:.4f}"
    )
    print(
        f"Mean OOD Acc:  @100%={summary['mean_acc_at_100']:.4f} | "
        f"@90%={summary['mean_acc_at_90']:.4f} | "
        f"@80%={summary['mean_acc_at_80']:.4f}"
    )
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()