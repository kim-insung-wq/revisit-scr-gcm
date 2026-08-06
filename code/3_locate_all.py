"""Step 3: locate all 32 partial products with the timing model, NCC, and CPA."""

from __future__ import annotations
import argparse
import matplotlib.pyplot as plt
import numpy as np
from common import (
    CONFIG, OUTPUT_DIR, cpa_validate_starts, detect_ncc_starts, load_dataset,
    load_json, save_figure, save_json, timing_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()

    _, traces, _, public_a = load_dataset()
    first_result = load_json(OUTPUT_DIR / "02_first_start.json")
    first_start = int(first_result["first_operation_start"])
    mean_trace = traces.mean(axis=0)

    model_rows = timing_model(first_start, CONFIG)
    ncc_rows, ncc_curve, ncc_reference = detect_ncc_starts(mean_trace, model_rows, CONFIG)
    refined_rows, cpa_deltas, cpa_global_scores = cpa_validate_starts(
        traces, public_a, ncc_rows, CONFIG
    )
    cut_starts = np.asarray([row["final_cut_start"] for row in refined_rows], dtype=np.int64)

    save_json(OUTPUT_DIR / "03_operation_starts.json", refined_rows)
    np.savez_compressed(
        OUTPUT_DIR / "03_localization_curves.npz",
        cut_starts=cut_starts,
        ncc_curve=ncc_curve,
        ncc_reference=ncc_reference,
        cpa_deltas=cpa_deltas,
        cpa_global_scores=cpa_global_scores,
    )

    print("Detected operation count :", len(refined_rows))
    print("Minimum selected NCC     :", min(row["ncc"] for row in refined_rows))
    print("Nonzero NCC corrections  :", sum(row["ncc_offset"] != 0 for row in refined_rows))
    print("Common EOR CPA phase     :", refined_rows[0]["cpa_global_phase"], "samples")

    fig, axes = plt.subplots(3, 1, figsize=(17, 11), constrained_layout=True)
    axes[0].plot(mean_trace, linewidth=0.55)
    axes[0].scatter(cut_starts, mean_trace[cut_starts], s=18)
    axes[0].set_xlim(max(0, cut_starts[0] - 300), min(mean_trace.size, cut_starts[-1] + 600))
    axes[0].set_xlabel("ADC sample")
    axes[0].set_ylabel("Mean ADC")
    axes[0].set_title("Mean waveform and 32 cut starts")
    axes[0].grid(alpha=0.25)

    axes[1].plot(ncc_curve, linewidth=0.65)
    axes[1].scatter([row["ncc_start"] for row in refined_rows], [row["ncc"] for row in refined_rows], s=18)
    axes[1].set_xlim(max(0, cut_starts[0] - 300), min(ncc_curve.size, cut_starts[-1] + 600))
    axes[1].set_xlabel("ADC sample")
    axes[1].set_ylabel("NCC")
    axes[1].set_title("Stride-1 sliding NCC")
    axes[1].grid(alpha=0.25)

    best_delta = cpa_deltas[int(np.argmax(cpa_global_scores))]
    axes[2].plot(cpa_deltas, cpa_global_scores, marker="o")
    axes[2].axvline(best_delta, linestyle="--")
    axes[2].set_xlabel("EOR leakage phase relative to NCC boundary")
    axes[2].set_ylabel("Mean absolute correlation")
    axes[2].set_title("Common EOR CPA phase validation")
    axes[2].grid(alpha=0.25)
    print("Figure:", save_figure("03_all_partial_products.png", args.show))


if __name__ == "__main__":
    main()
