"""Step 5: train the denoising-autoencoder ensemble and save latent vectors."""

from __future__ import annotations
import argparse
import matplotlib.pyplot as plt
import numpy as np
from common import (
    CONFIG, OUTPUT_DIR, row_normalize, save_figure, save_json,
    standardize_columns, train_autoencoder,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()

    input_path = OUTPUT_DIR / "04_training_x_raw.npy"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run: python code/4_extract.py")
    training_x_raw = np.load(input_path)
    training_x = row_normalize(training_x_raw)

    models = []
    losses = []
    latent_parts = []
    for seed in CONFIG["ae_seeds"]:
        print(f"Training autoencoder seed {seed}...")
        model, loss = train_autoencoder(training_x, CONFIG, int(seed))
        models.append(model)
        losses.append(loss)
        latent_parts.append(standardize_columns(model.encode(training_x)))

    latent = np.concatenate(latent_parts, axis=1)
    losses_array = np.stack(losses, axis=0)
    archive: dict[str, np.ndarray] = {"latent": latent, "losses": losses_array}
    for model_index, model in enumerate(models):
        for name in ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4"):
            archive[f"model_{model_index}_{name}"] = getattr(model, name)
    np.savez_compressed(OUTPUT_DIR / "05_autoencoder_ensemble.npz", **archive)
    np.save(OUTPUT_DIR / "05_latent_vectors.npy", latent)

    summary = {
        "normalized_input_shape": list(training_x.shape),
        "ensemble_latent_shape": list(latent.shape),
        "seeds": [int(seed) for seed in CONFIG["ae_seeds"]],
        "final_reconstruction_mse": [float(loss[-1]) for loss in losses],
    }
    save_json(OUTPUT_DIR / "05_autoencoder_summary.json", summary)
    print("Normalized input shape:", training_x.shape)
    print("Ensemble latent shape :", latent.shape)
    print("Final reconstruction MSE:", summary["final_reconstruction_mse"])

    plt.figure(figsize=(14, 5))
    for seed, loss in zip(CONFIG["ae_seeds"], losses):
        plt.plot(loss, linewidth=1.0, label=f"seed {seed}")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction MSE")
    plt.title("Denoising autoencoder training")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    print("Figure:", save_figure("05_autoencoder_training.png", args.show))


if __name__ == "__main__":
    main()
