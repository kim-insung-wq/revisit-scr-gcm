"""Step 6: cluster all 576 latent vectors, orient labels, and reconstruct H."""

from __future__ import annotations
import argparse
import matplotlib.pyplot as plt
import numpy as np
from common import (
    CONFIG, OUTPUT_DIR, kmeans_two, load_json, save_figure, save_json,
    silhouette, validate_clusters,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()

    latent_path = OUTPUT_DIR / "05_latent_vectors.npy"
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing {latent_path}. Run: python code/5_autoencoder.py")
    latent = np.load(latent_path)
    refined_rows = load_json(OUTPUT_DIR / "03_operation_starts.json")

    labels, centers, inertia = kmeans_two(latent)
    silhouette_value = silhouette(latent, labels)
    validation = validate_clusters(labels, refined_rows, CONFIG)
    trace_count = labels.size // len(refined_rows)
    label_map = labels.reshape(len(refined_rows), trace_count)
    orientation_flip = int(validation.get("selected_global_orientation_flip", 0))
    oriented_label_map = label_map ^ orientation_flip

    limb_hex = validation["recovered_internal_limbs_hex"]
    b0, b1, b2, b3 = [int(value, 16) for value in limb_hex]
    h_internal = b0 | (b1 << 32) | (b2 << 64) | (b3 << 96)
    h_standard = int(f"{h_internal:0128b}"[::-1], 2)

    result = {
        **validation,
        "kmeans_inertia": float(inertia),
        "silhouette": float(silhouette_value),
        "orientation_flip": orientation_flip,
        "h_internal_hex": f"{h_internal:032X}",
        "h_standard_hex": f"{h_standard:032X}",
    }
    save_json(OUTPUT_DIR / "06_recovered_h.json", result)
    np.savez_compressed(
        OUTPUT_DIR / "06_clustering_results.npz",
        labels=labels,
        centers=centers,
        label_map=label_map,
        oriented_label_map=oriented_label_map,
    )

    print("Cluster sizes      :", validation["cluster_0_size"], "/", validation["cluster_1_size"])
    print("K-means inertia    :", inertia)
    print("Silhouette         :", silhouette_value)
    print("Pair agreement     :", validation.get("pair_agreement_rate"))
    print("Karatsuba violations:", validation.get("strict_karatsuba_total_violations"))
    print("Orientation flip   :", orientation_flip)
    print("b0..b3             :", limb_hex)
    print("H_internal         :", f"{h_internal:032X}")
    print("H_standard         :", f"{h_standard:032X}")

    plt.figure(figsize=(9, 7))
    for cluster in (0, 1):
        mask = labels == cluster
        plt.scatter(latent[mask, 0], latent[mask, 1], s=18, alpha=0.7, label=f"cluster {cluster}")
    plt.xlabel("Ensemble latent coordinate 0")
    plt.ylabel("Ensemble latent coordinate 1")
    plt.title("Global K-means over 576 latent vectors")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    print("Figure:", save_figure("06_latent_clusters.png", args.show))

    plt.figure(figsize=(14, 7))
    plt.imshow(oriented_label_map, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    plt.axvline(8.5, linestyle="--", linewidth=1.0)
    plt.xlabel("Trace index")
    plt.ylabel("(l, m) position index")
    plt.title("Oriented cluster labels for all 576 subtraces")
    plt.colorbar(label="Recovered B bit")
    plt.tight_layout()
    print("Figure:", save_figure("06_oriented_labels.png", args.show))


if __name__ == "__main__":
    main()
