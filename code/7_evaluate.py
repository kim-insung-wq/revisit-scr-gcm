"""Step 7 (optional): compare the recovered labels and H with ground truth."""

from __future__ import annotations
import argparse
import matplotlib.pyplot as plt
import numpy as np
from common import (
    CONFIG, OUTPUT_DIR, ground_truth_from_h, load_json, save_figure, save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()

    cluster_path = OUTPUT_DIR / "06_clustering_results.npz"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Missing {cluster_path}. Run: python code/6_cluster_recover.py")
    cluster_data = np.load(cluster_path)
    oriented_label_map = cluster_data["oriented_label_map"]
    refined_rows = load_json(OUTPUT_DIR / "03_operation_starts.json")
    recovered = load_json(OUTPUT_DIR / "06_recovered_h.json")

    truth_map, truth_metadata = ground_truth_from_h(
        CONFIG["ground_truth_h_hex"],
        refined_rows,
        oriented_label_map.shape[1],
        int(CONFIG["pair_offset"]),
    )
    correct_map = oriented_label_map == truth_map
    recovered_limbs = recovered["recovered_internal_limbs_hex"]
    truth_limbs = truth_metadata["internal_limbs_b0_to_b3_hex"]

    result = {
        "ground_truth_h_hex": truth_metadata["standard_h_hex"],
        "recovered_h_hex": recovered["h_standard_hex"],
        "label_accuracy": float(correct_map.mean()),
        "correct_labels": int(correct_map.sum()),
        "total_labels": int(correct_map.size),
        "recovered_limbs_match": recovered_limbs == truth_limbs,
    }
    save_json(OUTPUT_DIR / "07_evaluation.json", result)
    for key, value in result.items():
        print(f"{key:24s}: {value}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].imshow(oriented_label_map, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    axes[0].set_title("Recovered oriented labels")
    axes[1].imshow(truth_map, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    axes[1].set_title("Ground-truth B bits")
    axes[2].imshow(oriented_label_map != truth_map, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("Mismatch map")
    for axis in axes:
        axis.axvline(8.5, linestyle="--", linewidth=1.0)
        axis.set_xlabel("Trace index")
        axis.set_ylabel("(l, m) position index")
    print("Figure:", save_figure("07_ground_truth_evaluation.png", args.show))


if __name__ == "__main__":
    main()
