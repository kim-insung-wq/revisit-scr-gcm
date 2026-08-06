"""Step 4: extract the 32 x 18 = 576 autoencoder subtraces."""

from __future__ import annotations
import argparse
import matplotlib.pyplot as plt
import numpy as np
from common import CONFIG, OUTPUT_DIR, load_dataset, load_json, save_figure, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()

    traces_raw, traces, _, _ = load_dataset()
    refined_rows = load_json(OUTPUT_DIR / "03_operation_starts.json")
    ae_window_samples = int(CONFIG["ae_window_samples"])

    full_crops: dict[str, np.ndarray] = {}
    ae_windows = []
    for row in refined_rows:
        start = int(row["final_cut_start"])
        operation_length = int(row["operation_length"])
        name = f"l_{row['l']}_m_{row['m']}"
        full_crops[name] = np.asarray(traces_raw[:, start : start + operation_length])
        ae_windows.append(
            np.asarray(traces[:, start : start + ae_window_samples], dtype=np.float64)
        )

    ae_windows_array = np.stack(ae_windows, axis=0)
    training_x_raw = ae_windows_array.reshape(-1, ae_window_samples)
    np.save(OUTPUT_DIR / "04_ae_windows.npy", ae_windows_array)
    np.save(OUTPUT_DIR / "04_training_x_raw.npy", training_x_raw)
    np.savez_compressed(OUTPUT_DIR / "04_full_operation_crops.npz", **full_crops)

    summary = {
        "ae_window_array_shape": list(ae_windows_array.shape),
        "flattened_trace_count": int(training_x_raw.shape[0]),
        "ae_input_length": int(training_x_raw.shape[1]),
    }
    save_json(OUTPUT_DIR / "04_extraction_summary.json", summary)
    for key, value in summary.items():
        print(f"{key:24s}: {value}")

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    for trace_index in range(18):
        axes[0].plot(full_crops["l_1_m_0"][trace_index], linewidth=0.6, alpha=0.65)
    axes[0].set_xlabel("Sample within complete operation")
    axes[0].set_ylabel("Raw ADC")
    axes[0].set_title("Complete operation windows: l=1, m=0")
    axes[0].grid(alpha=0.25)

    for trace_index in range(18):
        axes[1].plot(ae_windows_array[4, trace_index], linewidth=0.6, alpha=0.65)
    axes[1].set_xlabel(f"Sample within {ae_window_samples}-sample AE window")
    axes[1].set_ylabel("Raw ADC")
    axes[1].set_title("Autoencoder windows: l=1, m=0")
    axes[1].grid(alpha=0.25)
    print("Figure:", save_figure("04_extracted_subtraces.png", args.show))


if __name__ == "__main__":
    main()
