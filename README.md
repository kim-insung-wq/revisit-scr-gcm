# Reproduction Artifact for “Revisiting Side-Channel Resistance of GCM Implementations on AVR Microcontrollers”

This artifact reproduces the feature-extraction and autoencoder-based clustering attack described in **“Revisiting Side-Channel Resistance of GCM Implementations on AVR Microcontrollers.”**

The original paper collected traces from a KLA-SCARF AVR board equipped with an ATmega128. In this artifact, the same 32-bit Block-Comb multiplication protected by Dummy XOR and instruction-level atomicity (ILA) was measured using a **ChipWhisperer XMEGA target and CW-Husky**.

The acquisition settings are:

- target clock: 7.3728 MHz;
- Husky ADC: 12 bit, `adc_mul = 27` (approximately 199.1 MS/s);
- 16,000 samples per trace and 20 dB gain;
- rising-edge trigger on `TIO4`;
- SimpleSerial communication at 38,400 baud.

> **Source-code availability.** The protected multiplication implementation used in this reproduction is based on the algorithm presented by Seo and Kim, *“SCA-Resistant GCM Implementation on 8-bit AVR Microcontrollers,”* IEEE Access, vol. 7, pp. 103961–103978, 2019. The original multiplication source code is not redistributed with this artifact. Researchers seeking access should contact the authors of the original paper.

For one ciphertext block without AAD, GHASH performs two 128-bit multiplications. Each is decomposed into nine 32-bit Karatsuba multiplications, and each 32-bit multiplication contains 32 partial products. The dataset therefore yields

$2 \times 9 \times 32 = 576$

subtraces.

## Directory structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── run.py                         # run Steps 1-7 automatically
├── code/
│   ├── common.py
│   ├── 1_load.py
│   ├── 2_locate_first.py
│   ├── 3_locate_all.py
│   ├── 4_extract.py
│   ├── 5_autoencoder.py
│   ├── 6_cluster_recover.py
│   └── 7_evaluate.py
├── datasets/
│   ├── traces_raw_12bit.npy
│   └── public_values.json
└── outputs/                         # generated automatically
    ├── figures/
    ├── 01_dataset_summary.json
    ├── 02_first_start.json
    ├── 03_operation_starts.json
    ├── 04_ae_windows.npy
    ├── 05_latent_vectors.npy
    ├── 06_recovered_h.json
    └── 07_evaluation.json
```

The two input files must be placed directly under `datasets/`. All intermediate arrays, JSON summaries, recovered values, and figures are written under `outputs/`.

## Requirements

- Python 3.10 or later
- NumPy
- Matplotlib

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Run the complete pipeline (recommended)

Run the following command from the repository root:

```bash
python run.py
```

`run.py` executes Steps 1-7 in order. Each stage writes its intermediate results under `outputs/`, and the next stage reads those files. If any stage fails, the pipeline stops immediately instead of continuing with incomplete results.

To display all generated figures interactively while running:

```bash
python run.py --show
```

A partial range can also be re-run after the prerequisite outputs already exist. For example, to repeat only autoencoder training through evaluation:

```bash
python run.py --from-step 5 --to-step 7
```

## Execution order and individual stages

Run every command from the repository root. The individual scripts are connected through files under `outputs/`; they are separate processes rather than one in-memory Python process. Therefore, the stages should normally be executed in order.

### Step 1. Load and validate the dataset

```bash
python code/1_load.py
```

This script:

- loads `traces_raw_12bit.npy` and `public_values.json`;
- checks that the trace array contains 18 traces;
- checks that nine public 32-bit multiplicands are available;
- plots the first complete trace.

Main outputs:

```text
outputs/01_dataset_summary.json
outputs/figures/01_full_trace.png
```

### Step 2. Locate the first partial product with CPA

```bash
python code/2_locate_first.py
```

This script uses the Hamming weights of the public multiplicand bytes to identify the first four-EOR pattern and determine the first partial-product boundary.

Main outputs:

```text
outputs/02_first_start.json
outputs/02_first_detection.npz
outputs/figures/02_first_partial_product.png
```

### Step 3. Locate all 32 partial products with NCC and CPA

```bash
python code/3_locate_all.py
```

This script:

- predicts all 32 operation positions using the assembly timing model;
- refines each predicted position with stride-1 normalized cross-correlation;
- validates the common EOR leakage phase with CPA.

Main outputs:

```text
outputs/03_operation_starts.json
outputs/03_localization_curves.npz
outputs/figures/03_all_partial_products.png
```

### Step 4. Extract the 576 subtraces

```bash
python code/4_extract.py
```

The 32 detected boundaries are applied to all 18 captured 32-bit multiplications:

$$
32 \times 18 = 576.
$$

The complete operation windows are retained for inspection, while an 81-sample region from each operation is used as the autoencoder input.

Main outputs:

```text
outputs/04_ae_windows.npy                  # shape: (32, 18, 81)
outputs/04_training_x_raw.npy              # shape: (576, 81)
outputs/04_full_operation_crops.npz
outputs/04_extraction_summary.json
outputs/figures/04_extracted_subtraces.png
```
> **Remark on autoencoder hyperparameters.** Because the traces in this artifact were newly acquired using a ChipWhisperer XMEGA target and CW-Husky rather than taken directly from the original KLA-SCARF dataset, their sampling rate, trace length, noise characteristics, and temporal alignment differ from those reported in the original paper. Consequently, the subtrace length, network architecture, training parameters, and ensemble configuration were adjusted for the reproduced dataset. These changes adapt the autoencoder to the measurement characteristics of the new raw traces; they do not change the underlying attack procedure, which still consists of partial-product localization, subtrace extraction, unsupervised feature learning, global two-cluster classification, and consistency-based recovery of (H).

### Step 5. Train the denoising-autoencoder ensemble

```bash
python code/5_autoencoder.py
```

Each 81-sample subtrace is normalized independently. Five denoising autoencoders with architecture

```text
81 -> 32 -> 8 -> 32 -> 81
```

are trained with different random seeds. Their standardized latent representations are concatenated into one 40-dimensional vector per subtrace.

Main outputs:

```text
outputs/05_autoencoder_ensemble.npz
outputs/05_latent_vectors.npy              # shape: (576, 40)
outputs/05_autoencoder_summary.json
outputs/figures/05_autoencoder_training.png
```

### Step 6. Cluster the latent vectors and recover H

```bash
python code/6_cluster_recover.py
```

One global two-cluster K-means model is applied to all 576 latent vectors. Cluster orientation is determined without the ground-truth key by minimizing repeated-trace and Karatsuba consistency violations.

For internal 32-bit limbs \(b_0,b_1,b_2,b_3\), the nine multiplier operands are

$$
b_0,\ b_1,\ b_0\oplus b_1,\ b_2,\ b_3,\ b_2\oplus b_3,\
b_0\oplus b_2,\ b_1\oplus b_3,\ b_0\oplus b_1\oplus b_2\oplus b_3.
$$

The oriented labels recover the four limbs, which are combined and converted from the implementation bit order to the standard GHASH representation.

Main outputs:

```text
outputs/06_clustering_results.npz
outputs/06_recovered_h.json
outputs/figures/06_latent_clusters.png
outputs/figures/06_oriented_labels.png
```

### Step 7. Evaluate against the known H value (optional)

```bash
python code/7_evaluate.py
```

The configured ground-truth value of \(H\) is used only in this final evaluation step. It is not used for trace localization, autoencoder training, clustering, or cluster orientation.

Main outputs:

```text
outputs/07_evaluation.json
outputs/figures/07_ground_truth_evaluation.png
```

## Manual sequential execution

The following commands are equivalent to `python run.py`, provided that every command succeeds:

```bash
python code/1_load.py
python code/2_locate_first.py
python code/3_locate_all.py
python code/4_extract.py
python code/5_autoencoder.py
python code/6_cluster_recover.py
python code/7_evaluate.py
```

The connection between stages is:

```text
1_load.py
  -> outputs/01_dataset_summary.json

2_locate_first.py
  -> outputs/02_first_start.json

3_locate_all.py
  -> outputs/03_operation_starts.json

4_extract.py
  -> outputs/04_training_x_raw.npy

5_autoencoder.py
  -> outputs/05_latent_vectors.npy

6_cluster_recover.py
  -> outputs/06_recovered_h.json
     outputs/06_clustering_results.npz

7_evaluate.py
  -> outputs/07_evaluation.json
```

Thus, running a later stage by itself works only when its prerequisite files have already been generated. `run.py` handles this ordering automatically.

## Interactive figures

By default, figures are saved under `outputs/figures/` without blocking execution. To also display the figure produced by a stage, add `--show`:

```bash
python code/3_locate_all.py --show
```

## Re-running a stage

A stage may be re-run after changing parameters in `code/common.py`. Re-run every later stage whose input depends on the changed result. For example, after changing the localization parameters, execute Steps 2–7 again.
