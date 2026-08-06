"""Step 2: locate the first partial-product operation with public-A CPA."""

from __future__ import annotations
import argparse
import matplotlib.pyplot as plt
import numpy as np
from common import CONFIG, OUTPUT_DIR, load_dataset, locate_first_start, save_figure, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()

    _, traces, _, public_a = load_dataset()
    first_start, candidates, scores, cpa, clock_phase = locate_first_start(
        traces, public_a, CONFIG
    )
    mean_trace = traces.mean(axis=0)

    result = {
        "clock_phase": int(clock_phase),
        "first_operation_start": int(first_start),
        "best_structured_cpa": float(np.max(scores)),
    }
    save_json(OUTPUT_DIR / "02_first_start.json", result)
    np.savez_compressed(
        OUTPUT_DIR / "02_first_detection.npz",
        candidates=candidates,
        scores=scores,
        cpa=cpa,
    )

    print("Detected clock phase :", clock_phase)
    print("First operation start:", first_start)
    print("Best structured CPA  :", float(np.max(scores)))

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), constrained_layout=True)
    axes[0].plot(candidates, scores, marker="o", linewidth=1.0)
    axes[0].axvline(first_start, linestyle="--", label=f"selected = {first_start}")
    axes[0].set_xlabel("Candidate first-operation start")
    axes[0].set_ylabel("Structured CPA score")
    axes[0].set_title("Public-A CPA start detection")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(mean_trace, linewidth=0.55)
    axes[1].axvline(first_start, linestyle="--", label="first operation")
    axes[1].set_xlim(max(0, first_start - 300), min(mean_trace.size, first_start + 1600))
    axes[1].set_xlabel("ADC sample")
    axes[1].set_ylabel("Mean ADC")
    axes[1].set_title("Mean waveform near the first operation")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    print("Figure:", save_figure("02_first_partial_product.png", args.show))


if __name__ == "__main__":
    main()
