"""Analyze per-class geographic shift failures and generate summary figures."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ALL_REGIONS = (
    "Africa",
    "Americas",
    "EastAsia",
    "Europe",
    "SouthEastAsia",
    "WestAsia",
)

DEFAULT_EXPERIMENT_DIR = Path("outputs/experiments/linear_probe_resnet50")
DEFAULT_FIGURES_DIR = Path("outputs/figures")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for experiment analysis."""

    parser = argparse.ArgumentParser(
        description="Run per-class error analysis and plot GeoShift results."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Experiment directory with summary.json, reliability_summary.json, and fold dirs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Output directory for generated PNG charts.",
    )
    parser.add_argument(
        "--focus-region",
        type=str,
        default="Africa",
        choices=list(ALL_REGIONS),
        help="Held-out region for detailed per-class drop analysis (default: Africa).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of worst-dropping classes to report (default: 10).",
    )
    return parser.parse_args()


def compute_per_class_drops(fold_metrics: dict) -> list[dict]:
    """Compare in-distribution validation vs. OOD test accuracy for each class."""

    val_per_class = fold_metrics["validation_metrics"]["per_class"]
    test_per_class = fold_metrics["test_metrics"]["per_class"]

    rows: list[dict] = []
    for label, test_stats in test_per_class.items():
        val_stats = val_per_class[label]
        in_acc = float(val_stats["accuracy"])
        ood_acc = float(test_stats["accuracy"])
        rows.append(
            {
                "class": label,
                "in_dist_acc": round(in_acc, 4),
                "ood_acc": round(ood_acc, 4),
                "drop": round(in_acc - ood_acc, 4),
                "val_support": int(val_stats["support"]),
                "ood_support": int(test_stats["support"]),
            }
        )

    rows.sort(key=lambda item: item["drop"], reverse=True)
    return rows


def plot_regional_gap(summary: dict, output_path: Path) -> None:
    """Plot in-distribution vs. held-out OOD accuracy across all 6 regions."""

    folds = summary["folds"]
    regions = [f["region"] for f in folds]
    in_dist = [f["in_dist_val_accuracy"] * 100.0 for f in folds]
    ood = [f["ood_test_accuracy"] * 100.0 for f in folds]

    x = np.arange(len(regions))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=150)
    bars_in = ax.bar(x - width / 2, in_dist, width, label="In-Dist Val", color="#3b82f6")
    bars_ood = ax.bar(x + width / 2, ood, width, label="Held-Out OOD Test", color="#ef4444")

    for bar in bars_in:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars_ood:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        f"GeoShift Geographic Generalization ({summary['backbone']} Linear Probe)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylim(75.0, 98.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_selective_prediction(experiment_dir: Path, output_path: Path) -> None:
    """Plot OOD accuracy-vs-coverage curves for all 6 held-out regions."""

    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=150)

    for region in ALL_REGIONS:
        rel_path = experiment_dir / f"heldout_{region}" / "reliability.json"
        if not rel_path.exists():
            continue
        rel = json.loads(rel_path.read_text(encoding="utf-8"))
        curve = rel["ood_selective_prediction"]
        coverages = [pt["actual_coverage"] * 100.0 for pt in curve]
        accuracies = [pt["accuracy"] * 100.0 for pt in curve]
        ax.plot(coverages, accuracies, marker="o", linewidth=2, label=region)

    ax.set_xlabel("Coverage (% of samples accepted)")
    ax.set_ylabel("OOD Accuracy (%)")
    ax.set_title("Selective Prediction: OOD Accuracy vs. Coverage (Calibrated)")
    ax.invert_xaxis()
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    summary_path = args.experiment_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json at {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    focus_metrics_path = (
        args.experiment_dir / f"heldout_{args.focus_region}" / "metrics.json"
    )
    if not focus_metrics_path.exists():
        raise FileNotFoundError(f"Missing fold metrics at {focus_metrics_path}")

    focus_metrics = json.loads(focus_metrics_path.read_text(encoding="utf-8"))
    class_drops = compute_per_class_drops(focus_metrics)

    analysis_path = args.experiment_dir / f"error_analysis_{args.focus_region}.json"
    analysis_path.write_text(
        json.dumps(
            {
                "focus_region": args.focus_region,
                "worst_drops": class_drops[: args.top_n],
                "best_generalizing": list(reversed(class_drops[-5:])),
                "all_classes": class_drops,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    gap_fig = args.figures_dir / "regional_generalization_gap.png"
    sel_fig = args.figures_dir / "selective_prediction_curves.png"
    plot_regional_gap(summary, gap_fig)
    plot_selective_prediction(args.experiment_dir, sel_fig)

    print(f"GeoShift | Per-Class Error Analysis (Held-out: {args.focus_region})")
    print(f"{'Class':22} | {'In-Dist':>8} | {'OOD':>8} | {'Drop':>8} | {'OOD N':>6}")
    print("-" * 64)
    for row in class_drops[: args.top_n]:
        print(
            f"{row['class']:22} | "
            f"{row['in_dist_acc'] * 100:7.2f}% | "
            f"{row['ood_acc'] * 100:7.2f}% | "
            f"{row['drop'] * 100:+7.2f}% | "
            f"{row['ood_support']:6d}"
        )

    print(f"\nSaved analysis: {analysis_path}")
    print(f"Saved chart:    {gap_fig}")
    print(f"Saved chart:    {sel_fig}")


if __name__ == "__main__":
    main()