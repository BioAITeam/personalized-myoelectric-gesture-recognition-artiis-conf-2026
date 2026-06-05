from __future__ import annotations

import importlib.metadata as metadata
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reproducibility import DEFAULT_SEED, configure_reproducibility


PACKAGES = [
    "tensorflow",
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "notebook",
    "ipykernel",
    "nbconvert",
    "openpyxl",
    "pillow",
    "pypdfium2",
]


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def nvidia_smi() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "not detected"
    except Exception:
        return "not detected"


def main() -> int:
    state = configure_reproducibility(DEFAULT_SEED)
    print(f"Python: {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    print(f"Seed: {state.seed}")
    print(f"Deterministic TensorFlow ops: {state.deterministic_ops}")
    print()
    print("Packages:")
    for package in PACKAGES:
        print(f"  {package}: {package_version(package)}")
    print()
    print(f"NVIDIA GPU: {nvidia_smi()}")

    try:
        import tensorflow as tf

        devices = tf.config.list_physical_devices()
        gpus = tf.config.list_physical_devices("GPU")
        print(f"TensorFlow devices: {[device.name for device in devices]}")
        print(f"TensorFlow GPUs: {[gpu.name for gpu in gpus]}")
    except Exception as exc:
        print(f"TensorFlow check failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
