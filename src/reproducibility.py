from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

DEFAULT_SEED = 42


@dataclass(frozen=True)
class ReproducibilityState:
    seed: int
    deterministic_ops: bool
    tensorflow_configured: bool


def configure_reproducibility(
    seed: int = DEFAULT_SEED,
    *,
    deterministic_ops: bool = True,
    include_tensorflow: bool = True,
) -> ReproducibilityState:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_ops:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    random.seed(seed)
    np.random.seed(seed)

    tensorflow_configured = False
    if include_tensorflow:
        try:
            import tensorflow as tf

            tf.keras.utils.set_random_seed(seed)
            if deterministic_ops:
                try:
                    tf.config.experimental.enable_op_determinism()
                except Exception:
                    pass
            tensorflow_configured = True
        except ImportError:
            tensorflow_configured = False

    return ReproducibilityState(
        seed=seed,
        deterministic_ops=deterministic_ops,
        tensorflow_configured=tensorflow_configured,
    )
