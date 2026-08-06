"""Run the complete GHASH side-channel reproduction pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parent

STEPS = [
    (1, "Load and validate the dataset", "1_load.py"),
    (2, "Locate the first partial product", "2_locate_first.py"),
    (3, "Locate all 32 partial products", "3_locate_all.py"),
    (4, "Extract the 576 subtraces", "4_extract.py"),
    (5, "Train the autoencoder ensemble", "5_autoencoder.py"),
    (6, "Cluster latent vectors and recover H", "6_cluster_recover.py"),
    (7, "Evaluate the recovered result", "7_evaluate.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Steps 1-7 in order. Each stage writes its intermediate results "
            "to outputs/, and the next stage reads those files."
        )
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also display each stage's figures interactively",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        choices=range(1, 8),
        metavar="N",
        help="start from step N (default: 1)",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        default=7,
        choices=range(1, 8),
        metavar="N",
        help="stop after step N (default: 7)",
    )
    args = parser.parse_args()
    if args.from_step > args.to_step:
        parser.error("--from-step must be less than or equal to --to-step")
    return args


def main() -> int:
    args = parse_args()
    selected_steps = [
        step for step in STEPS if args.from_step <= step[0] <= args.to_step
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print("=" * 72, flush=True)
    print("GHASH side-channel reproduction pipeline", flush=True)
    print(f"Repository : {ROOT_DIR}", flush=True)
    print(f"Steps      : {args.from_step} through {args.to_step}", flush=True)
    print("=" * 72, flush=True)

    for number, description, filename in selected_steps:
        script_path = ROOT_DIR / "code" / filename
        command = [sys.executable, str(script_path)]
        if args.show:
            command.append("--show")

        print(flush=True)
        print("-" * 72, flush=True)
        print(f"[Step {number}/7] {description}", flush=True)
        print("Command:", " ".join(command), flush=True)
        print("-" * 72, flush=True)

        result = subprocess.run(command, cwd=ROOT_DIR, env=env, check=False)
        if result.returncode != 0:
            print(flush=True)
            print(
                f"Pipeline stopped: Step {number} failed "
                f"with exit code {result.returncode}.",
                file=sys.stderr,
                flush=True,
            )
            return result.returncode

    print(flush=True)
    print("=" * 72, flush=True)
    print("Pipeline completed successfully.", flush=True)
    print(f"Results: {ROOT_DIR / 'outputs'}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
