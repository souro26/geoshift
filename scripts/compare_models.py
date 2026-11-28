"""Compare ResNet-50 (CNN) vs. Swin-Tiny (ViT) on held-out geographic shift."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESNET_DIR = Path("outputs/experiments/linear_probe_resnet50")
DEFAULT_SWIN_DIR = Path("outputs/experiments/linear_probe_swin_t")
DEFAULT_OUTPUT_PATH = Path("outputs/figures/cnn_vs_vit_africa.png")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model comparison chart."""

    parser = argparse.ArgumentParser(
        description="Plot ResNet-50 vs. Swin-Tiny per-class accuracy on Africa."
    )
    parser.add_argument("--resnet-dir", type=Path, default=DEFAULT_RESNET_DIR)
    parser.add_argument("--swin-dir", type=Path, default=DEFAULT_SWIN_DIR)
    parser.add_argument("--region", type=str, default="Africa")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    resnet_path = args.resnet_dir / f"error_analysis_{args.region}.json"
    swin_path = args.swin_dir / f"error_analysis_{args.region}.json"

    if not resnet_path.exists():
        raise FileNotFoundError(f"Missing ResNet analysis: {resnet_path}")
    if not swin_path.exists():
        raise FileNotFoundError(f"Missing Swin analysis: {swin_path}")

    resnet_data = json.loads(resnet_path.read_text(encoding="utf-8"))
    swin_data = json.loads(swin_path.read_text(encoding="utf-8"))

    swin_by_class = {row["class"]: row for row in swin_data["all_classes"]}
    top_rows = resnet_data["worst_drops"][: args.top_n]

    classes = [row["class"] for row in top_rows]
    resnet_ood = [row["ood_acc"] * 100.0 for row in top_rows]
    swin_ood = [swin_by_class[c]["ood_acc"] * 100.0 for c in classes]

    y = np.arange(len(classes))
    height = 0.36

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=150)
    bars_r = ax.barh(y - height / 2, resnet_ood, height, label="ResNet-50 (CNN)", color="#2563eb")
    bars_s = ax.barh(y + height / 2, swin_ood, height, label="Swin-Tiny (ViT)", color="#f97316")

    for bar in bars_r:
        w = bar.get_width()
        ax.annotate(f"{w:.1f}%", xy=(w, bar.get_y() + bar.get_height() / 2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    for bar in bars_s:
        w = bar.get_width()
        ax.annotate(f"{w:.1f}%", xy=(w, bar.get_y() + bar.get_height() / 2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel("Held-Out OOD Accuracy (%)")
    ax.set_xlim(40.0, 105.0)
    ax.set_title(f"CNN vs. Vision Transformer on Hardest Held-Out Classes ({args.region})")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)

    print(f"Saved comparison chart to: {args.output}")


if __name__ == "__main__":
    main()