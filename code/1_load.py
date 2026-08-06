"""Step 1: load and validate the trace dataset."""

from __future__ import annotations
import argparse
import matplotlib.pyplot as plt
from common import CONFIG, DATASET_DIR, OUTPUT_DIR, load_dataset, save_figure, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()

    traces_raw, traces, public_json, public_a = load_dataset()
    summary = {
        "dataset_directory": str(DATASET_DIR),
        "trace_shape": list(traces.shape),
        "trace_dtype": str(traces_raw.dtype),
        "ciphertext_c1": public_json.get("c1_hex"),
        "length_block": public_json.get("length_block_hex"),
        "public_a_count": int(public_a.size),
        "samples_per_cycle": int(CONFIG["samples_per_cycle"]),
    }
    save_json(OUTPUT_DIR / "01_dataset_summary.json", summary)

    for key, value in summary.items():
        print(f"{key:20s}: {value}")

    plt.figure(figsize=(16, 4))
    plt.plot(traces[0], linewidth=0.55)
    plt.xlabel("ADC sample")
    plt.ylabel("Raw 12-bit ADC")
    plt.title("Full trace 0")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    print("Figure:", save_figure("01_full_trace.png", args.show))


if __name__ == "__main__":
    main()
