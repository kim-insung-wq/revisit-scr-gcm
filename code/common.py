"""Shared configuration and utilities for the GHASH SCA reproduction artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "datasets"
TRACES_PATH = DATASET_DIR / "traces_raw_12bit.npy"
PUBLIC_VALUES_PATH = DATASET_DIR / "public_values.json"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

CONFIG: dict[str, Any] = {
    "public_a_key": "public_stage1_a32_hex",
    # Timing
    "samples_per_cycle": 27,
    "first_start_cycle_range": [45, 85],
    "outer_count": 8,
    "inner_count": 4,
    "l0_inner_cycles": 9,
    "regular_inner_cycles": 10,
    "l0_outer_cycles": 41,
    "regular_outer_cycles": 45,
    # NCC / CPA
    "ncc_reference": [1, 0],
    "ncc_search_radius": 13,
    "cpa_refine_radius": 13,
    "cpa_first_eor_cycle": 3,
    # Autoencoder
    "ae_window_samples": 81,
    "ae_hidden_nodes": 32,
    "ae_latent_nodes": 8,
    "ae_epochs": 700,
    "ae_learning_rate": 0.004,
    "ae_l1_or_l2": 1e-5,
    "ae_noise_std": 0.025,
    "ae_seeds": [17, 29, 43, 71, 101],
    # Structural validation
    "pair_offset": 9,
    "karatsuba_pair_count": 9,
    # Evaluation only; never used for localization, training, or orientation.
    "ground_truth_h_hex": "E8D8C8352E2C500ABDEBCF932E417E98",
}


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prerequisite output: {path}\n"
            "Run the preceding README step first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(name: str, show: bool = False) -> Path:
    ensure_output_dirs()
    path = FIGURE_DIR / name
    plt.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return path


def load_dataset() -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    if not TRACES_PATH.exists():
        raise FileNotFoundError(
            f"Missing trace file: {TRACES_PATH}\n"
            "Place traces_raw_12bit.npy in the dataset directory shown in README.md."
        )
    if not PUBLIC_VALUES_PATH.exists():
        raise FileNotFoundError(
            f"Missing public-values file: {PUBLIC_VALUES_PATH}\n"
            "Place public_values.json in the dataset directory shown in README.md."
        )

    traces_raw = np.load(TRACES_PATH, mmap_mode="r")
    traces = np.asarray(traces_raw, dtype=np.float64)
    public_json = json.loads(PUBLIC_VALUES_PATH.read_text(encoding="utf-8"))
    try:
        public_a = np.asarray(
            [int(value, 16) for value in public_json[CONFIG["public_a_key"]]],
            dtype=np.uint64,
        )
    except KeyError as exc:
        raise KeyError(
            f"public_values.json does not contain {CONFIG['public_a_key']!r}"
        ) from exc

    if traces.ndim != 2:
        raise ValueError(f"Expected a 2-D trace array, got {traces.shape}")
    if traces.shape[0] != 18:
        raise ValueError(f"Expected 18 full traces, got {traces.shape[0]}")
    if public_a.size != 9:
        raise ValueError(f"Expected 9 public A values, got {public_a.size}")

    return traces_raw, traces, public_json, public_a


def pearson_models(models: np.ndarray, traces: np.ndarray) -> np.ndarray:
    models = np.asarray(models, dtype=np.float64)
    traces = np.asarray(traces, dtype=np.float64)
    centered_models = models - models.mean(axis=1, keepdims=True)
    centered_traces = traces - traces.mean(axis=0, keepdims=True)
    numerator = centered_models @ centered_traces
    denominator = np.linalg.norm(centered_models, axis=1)[:, None] * np.linalg.norm(
        centered_traces, axis=0
    )[None, :]
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )


def pearson_pair(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    y = np.asarray(y, dtype=np.float64) - np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


def detect_clock_phase(mean_trace: np.ndarray, period: int) -> tuple[int, np.ndarray]:
    scores = np.zeros(period, dtype=np.float64)
    for phase in range(period):
        indices = np.arange(phase + 20 * period, mean_trace.size - 2, period)
        if indices.size:
            scores[phase] = np.mean(
                np.abs(
                    mean_trace[indices]
                    - 0.5 * (mean_trace[indices - 2] + mean_trace[indices + 2])
                )
            )
    return int(np.argmax(scores)), scores


def sliding_ncc(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return Pearson NCC for every one-sample window start."""
    signal = np.asarray(signal, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    length = reference.size
    centered_reference = reference - reference.mean()
    ref_norm = np.linalg.norm(centered_reference)
    numerator = np.correlate(signal, centered_reference, mode="valid")
    cumulative = np.concatenate(([0.0], np.cumsum(signal)))
    cumulative2 = np.concatenate(([0.0], np.cumsum(signal * signal)))
    window_sum = cumulative[length:] - cumulative[:-length]
    window_sum2 = cumulative2[length:] - cumulative2[:-length]
    window_norm = np.sqrt(
        np.maximum(window_sum2 - window_sum * window_sum / length, 0.0)
    )
    denominator = ref_norm * window_norm
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def timing_model(first_start: int, config: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    period = int(config["samples_per_cycle"])
    outer_count = int(config["outer_count"])
    inner_count = int(config["inner_count"])
    l0_outer = int(config["l0_outer_cycles"])
    regular_outer = int(config["regular_outer_cycles"])
    rows: list[tuple[int, int, int, int]] = []
    for l_value in range(outer_count):
        if l_value == 0:
            l_start = first_start
            inner_cycles = int(config["l0_inner_cycles"])
        else:
            l_start = first_start + (l0_outer + regular_outer * (l_value - 1)) * period
            inner_cycles = int(config["regular_inner_cycles"])
        for m_value in range(inner_count):
            start = l_start + m_value * inner_cycles * period
            rows.append((l_value, m_value, start, inner_cycles * period))
    return rows


def eor_models(public_a: np.ndarray, l_value: int) -> np.ndarray:
    byte_count = 4 if l_value == 0 else 5
    shifted = np.asarray(public_a, dtype=np.uint64) << np.uint64(l_value)
    return np.asarray(
        [
            [
                int((value >> np.uint64(8 * byte_index)) & np.uint64(0xFF)).bit_count()
                for value in shifted
            ]
            for byte_index in range(byte_count)
        ],
        dtype=np.float64,
    )


def locate_first_start(
    traces: np.ndarray, public_a: np.ndarray, config: dict[str, Any]
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:
    period = int(config["samples_per_cycle"])
    known_traces = traces[: public_a.size]
    models = eor_models(public_a, l_value=0)
    cpa = pearson_models(models, known_traces)
    clock_phase, _ = detect_clock_phase(traces.mean(axis=0), period)
    first_cycle, last_cycle = [int(value) for value in config["first_start_cycle_range"]]
    candidates = np.asarray(
        [clock_phase + cycle * period for cycle in range(first_cycle, last_cycle + 1)],
        dtype=np.int64,
    )
    eor_first = int(config["cpa_first_eor_cycle"])
    scores = np.asarray(
        [
            np.nanmean(
                [abs(cpa[q, start + (eor_first + q) * period]) for q in range(models.shape[0])]
            )
            for start in candidates
        ]
    )
    return int(candidates[np.nanargmax(scores)]), candidates, scores, cpa, clock_phase


def detect_ncc_starts(
    mean_trace: np.ndarray,
    model_rows: list[tuple[int, int, int, int]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    ref_l, ref_m = [int(value) for value in config["ncc_reference"]]
    reference_row = next(row for row in model_rows if row[:2] == (ref_l, ref_m))
    reference_start, reference_length = reference_row[2], reference_row[3]
    reference = mean_trace[reference_start : reference_start + reference_length]
    curve = sliding_ncc(mean_trace, reference)
    radius = int(config["ncc_search_radius"])
    detected_rows: list[dict[str, Any]] = []
    for l_value, m_value, expected, length in model_rows:
        lo = max(0, expected - radius)
        hi = min(curve.size, expected + radius + 1)
        location = lo + int(np.argmax(curve[lo:hi]))
        detected_rows.append(
            {
                "l": l_value,
                "m": m_value,
                "multiplier_bit": 8 * m_value + l_value,
                "timing_model_start": expected,
                "ncc_start": location,
                "ncc_offset": location - expected,
                "ncc": float(curve[location]),
                "operation_length": length,
            }
        )
    return detected_rows, curve, reference


def cpa_validate_starts(
    traces: np.ndarray,
    public_a: np.ndarray,
    ncc_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    period = int(config["samples_per_cycle"])
    radius = int(config["cpa_refine_radius"])
    eor_first = int(config["cpa_first_eor_cycle"])
    known = traces[: public_a.size]
    deltas = np.arange(-radius, radius + 1, dtype=np.int64)
    score_matrix = np.zeros((len(ncc_rows), deltas.size), dtype=np.float64)

    for row_index, row in enumerate(ncc_rows):
        models = eor_models(public_a, int(row["l"]))
        for delta_index, delta in enumerate(deltas):
            correlations = []
            for byte_index, model in enumerate(models):
                location = int(row["ncc_start"] + delta + (eor_first + byte_index) * period)
                correlations.append(abs(pearson_pair(model, known[:, location])))
            score_matrix[row_index, delta_index] = float(np.nanmean(correlations))

    global_scores = np.nanmean(score_matrix, axis=0)
    global_delta = int(deltas[int(np.nanargmax(global_scores))])
    refined_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(ncc_rows):
        local_delta = int(deltas[int(np.nanargmax(score_matrix[row_index]))])
        cpa_supports_ncc = abs(local_delta) <= 2 or abs(local_delta - global_delta) <= 2
        final_start = int(row["ncc_start"])
        refined = dict(row)
        refined.update(
            {
                "cpa_local_phase": local_delta,
                "cpa_local_score": float(np.max(score_matrix[row_index])),
                "cpa_global_phase": global_delta,
                "cpa_score_at_global_phase": float(
                    score_matrix[row_index, int(np.where(deltas == global_delta)[0][0])]
                ),
                "cpa_validates_ncc": int(cpa_supports_ncc),
                "final_cut_start": final_start,
                "final_offset_from_model": final_start - int(row["timing_model_start"]),
            }
        )
        refined_rows.append(refined)
    return refined_rows, deltas, global_scores


def row_normalize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return np.divide(x - mean, std, out=np.zeros_like(x), where=std > 0)


@dataclass
class Autoencoder:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    w4: np.ndarray
    b4: np.ndarray

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        latent = self.encode(x)
        return np.tanh(latent @ self.w3 + self.b3) @ self.w4 + self.b4


def xavier(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=(fan_in, fan_out))


def train_autoencoder(
    x: np.ndarray, config: dict[str, Any], seed: int
) -> tuple[Autoencoder, np.ndarray]:
    input_size = x.shape[1]
    hidden = int(config["ae_hidden_nodes"])
    latent = int(config["ae_latent_nodes"])
    epochs = int(config["ae_epochs"])
    learning_rate = float(config["ae_learning_rate"])
    weight_decay = float(config["ae_l1_or_l2"])
    noise_std = float(config["ae_noise_std"])
    rng = np.random.default_rng(seed)
    model = Autoencoder(
        w1=xavier(rng, input_size, hidden),
        b1=np.zeros(hidden),
        w2=xavier(rng, hidden, latent),
        b2=np.zeros(latent),
        w3=xavier(rng, latent, hidden),
        b3=np.zeros(hidden),
        w4=xavier(rng, hidden, input_size),
        b4=np.zeros(input_size),
    )
    names = ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4")
    first = {name: np.zeros_like(getattr(model, name)) for name in names}
    second = {name: np.zeros_like(getattr(model, name)) for name in names}
    losses = np.zeros(epochs, dtype=np.float64)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    for epoch in range(1, epochs + 1):
        noisy = x + rng.normal(0.0, noise_std, size=x.shape)
        h1 = np.tanh(noisy @ model.w1 + model.b1)
        z = np.tanh(h1 @ model.w2 + model.b2)
        h3 = np.tanh(z @ model.w3 + model.b3)
        output = h3 @ model.w4 + model.b4
        error = output - x
        losses[epoch - 1] = float(np.mean(error * error))
        d_output = 2.0 * error / error.size
        gradients: dict[str, np.ndarray] = {
            "w4": h3.T @ d_output + weight_decay * model.w4,
            "b4": d_output.sum(axis=0),
        }
        d_h3 = (d_output @ model.w4.T) * (1.0 - h3 * h3)
        gradients["w3"] = z.T @ d_h3 + weight_decay * model.w3
        gradients["b3"] = d_h3.sum(axis=0)
        d_z = (d_h3 @ model.w3.T) * (1.0 - z * z)
        gradients["w2"] = h1.T @ d_z + weight_decay * model.w2
        gradients["b2"] = d_z.sum(axis=0)
        d_h1 = (d_z @ model.w2.T) * (1.0 - h1 * h1)
        gradients["w1"] = noisy.T @ d_h1 + weight_decay * model.w1
        gradients["b1"] = d_h1.sum(axis=0)
        for name in names:
            gradient = np.clip(gradients[name], -1.0, 1.0)
            first[name] = beta1 * first[name] + (1.0 - beta1) * gradient
            second[name] = beta2 * second[name] + (1.0 - beta2) * gradient * gradient
            m_hat = first[name] / (1.0 - beta1**epoch)
            v_hat = second[name] / (1.0 - beta2**epoch)
            setattr(
                model,
                name,
                getattr(model, name)
                - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon),
            )
    return model, losses


def standardize_columns(x: np.ndarray) -> np.ndarray:
    std = x.std(axis=0, keepdims=True)
    return np.divide(
        x - x.mean(axis=0, keepdims=True),
        std,
        out=np.zeros_like(x),
        where=std > 0,
    )


def kmeans_two(
    x: np.ndarray, seed: int = 2026, restarts: int = 64
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(restarts):
        centers = x[rng.choice(x.shape[0], size=2, replace=False)].copy()
        labels = np.full(x.shape[0], -1, dtype=np.int64)
        for _ in range(100):
            distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            if not all(np.any(labels == cluster) for cluster in (0, 1)):
                break
            centers = np.stack([x[labels == cluster].mean(axis=0) for cluster in (0, 1)])
        if not all(np.any(labels == cluster) for cluster in (0, 1)):
            continue
        inertia = float(np.sum((x - centers[labels]) ** 2))
        if best is None or inertia < best[2]:
            best = labels.copy(), centers.copy(), inertia
    if best is None:
        raise RuntimeError("K-means failed")
    return best


def silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    distances = np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
    values = []
    for index in range(x.shape[0]):
        same = np.flatnonzero(labels == labels[index])
        same = same[same != index]
        other = np.flatnonzero(labels != labels[index])
        if not same.size or not other.size:
            values.append(0.0)
            continue
        a = float(distances[index, same].mean())
        b = float(distances[index, other].mean())
        values.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(np.mean(values))


def validate_clusters(
    labels: np.ndarray, starts: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    position_count = len(starts)
    trace_count = labels.size // position_count
    label_map = labels.reshape(position_count, trace_count)
    pair_offset = int(config.get("pair_offset", 0))
    result: dict[str, Any] = {
        "cluster_0_size": int(np.sum(labels == 0)),
        "cluster_1_size": int(np.sum(labels == 1)),
    }
    if pair_offset > 0 and trace_count >= 2 * pair_offset:
        pair_matches = label_map[:, :pair_offset] == label_map[:, pair_offset : 2 * pair_offset]
        result.update(
            {
                "pair_agreement_count": int(pair_matches.sum()),
                "pair_count": int(pair_matches.size),
                "pair_agreement_rate": float(pair_matches.mean()),
                "perfect_pair_positions": int(np.sum(pair_matches.sum(axis=1) == pair_offset)),
            }
        )

    karatsuba_count = int(config.get("karatsuba_pair_count", 0))
    if karatsuba_count == 9 and trace_count >= 9:
        raw_violations = []
        complement_violations = []
        for row in label_map[:, :9]:
            expected = np.asarray(
                [
                    row[0], row[1], row[0] ^ row[1], row[3], row[4], row[3] ^ row[4],
                    row[0] ^ row[3], row[1] ^ row[4],
                    row[0] ^ row[1] ^ row[3] ^ row[4],
                ]
            )
            raw_violations.append(int(np.sum(expected != row)))
            inverse = 1 - row
            inverse_expected = np.asarray(
                [
                    inverse[0], inverse[1], inverse[0] ^ inverse[1], inverse[3], inverse[4],
                    inverse[3] ^ inverse[4], inverse[0] ^ inverse[3],
                    inverse[1] ^ inverse[4],
                    inverse[0] ^ inverse[1] ^ inverse[3] ^ inverse[4],
                ]
            )
            complement_violations.append(int(np.sum(inverse_expected != inverse)))
        raw_total = int(np.sum(raw_violations))
        complement_total = int(np.sum(complement_violations))
        orientation_flip = int(complement_total < raw_total)
        oriented_map = label_map ^ orientation_flip
        selected_violations = complement_violations if orientation_flip else raw_violations
        limbs = [0, 0, 0, 0]
        for position, row in enumerate(oriented_map[:, :9]):
            bit_index = int(starts[position]["multiplier_bit"])
            for limb_index, pair_index in enumerate((0, 1, 3, 4)):
                limbs[limb_index] |= int(row[pair_index]) << bit_index
        result.update(
            {
                "raw_label_karatsuba_total_violations": raw_total,
                "globally_complemented_total_violations": complement_total,
                "selected_global_orientation_flip": orientation_flip,
                "strict_karatsuba_total_violations": int(np.sum(selected_violations)),
                "strict_karatsuba_zero_violation_positions": int(
                    np.sum(np.asarray(selected_violations) == 0)
                ),
                "recovered_internal_limbs_hex": [f"{value:08X}" for value in limbs],
            }
        )
    return result


def ground_truth_from_h(
    h_hex: str,
    starts: list[dict[str, Any]],
    trace_count: int,
    pair_offset: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    standard_h = int(h_hex, 16)
    if standard_h.bit_length() > 128:
        raise ValueError("ground_truth_h_hex must fit in 128 bits")
    internal_h = int(f"{standard_h:0128b}"[::-1], 2)
    limbs = [(internal_h >> (32 * index)) & 0xFFFFFFFF for index in range(4)]
    b0, b1, b2, b3 = limbs
    pair_values = [
        b0, b1, b0 ^ b1, b2, b3, b2 ^ b3,
        b0 ^ b2, b1 ^ b3, b0 ^ b1 ^ b2 ^ b3,
    ]
    if pair_offset != 9 or trace_count != 18:
        raise ValueError(
            "Ground-truth evaluation expects 9 Karatsuba pairs repeated over 18 traces"
        )
    truth = np.zeros((len(starts), trace_count), dtype=np.int64)
    for position, row in enumerate(starts):
        bit_index = int(row["multiplier_bit"])
        for trace_index in range(trace_count):
            pair_index = trace_index % pair_offset
            truth[position, trace_index] = (pair_values[pair_index] >> bit_index) & 1
    metadata = {
        "standard_h_hex": f"{standard_h:032X}",
        "internal_h_hex": f"{internal_h:032X}",
        "internal_limbs_b0_to_b3_hex": [f"{value:08X}" for value in limbs],
        "karatsuba_b0_to_b8_hex": [f"{value:08X}" for value in pair_values],
    }
    return truth, metadata
