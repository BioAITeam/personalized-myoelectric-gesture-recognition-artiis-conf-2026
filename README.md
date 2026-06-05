# Personalized Hand Gesture Recognition from Forearm sEMG Signals Using Deep Learning for Myoelectric Control

This repository contains a reproducible workflow for personalized hand gesture recognition using forearm surface electromyography (sEMG) signals and MLP/CNN1D deep learning models.

The intended use is straightforward: clone the code from GitHub, download the Zenodo ZIP containing `Data/` and `Outputs/`, extract it at the project root, and run the notebooks in order without moving files around.

## Repository Scope

The study uses anonymized sEMG recordings from 3 participants (`0007`, `0008`, `0010`), 6 hand gestures (`g1` to `g6`), and 7 useful sEMG channels. Each participant-gesture combination is represented by one clean 20 s segment.

Gestures:

| Code | Gesture |
|---|---|
| `g1` | Fist |
| `g2` | Open fingers |
| `g3` | Wrist flexion inward |
| `g4` | Wrist extension outward |
| `g5` | Pinch |
| `g6` | Pointing |


Sensors:

| Alias | Sensor |
|---|---|
| `S1` | Right Brachioradialis |
| `S2` | Right Dorsal interossei |
| `S3` | Right Extensor carpi radialis longus |
| `S4` | Right Extensor carpi ulnaris |
| `S5` | Right Extensor digitorum |
| `S6` | Right Flexor carpi radialis |
| `S7` | Right Palmaris longus |

## Data and Outputs

GitHub contain the code, notebooks, documentation, and lightweight support files. Zenodo contain the heavy folders:

| Folder | Purpose | Expected location |
|---|---|---|
| `Data/` | Anonymized recordings, consolidated 20 s signals, and ML window matrices | Project root |
| `Outputs/` | Figures, metrics, confusion matrices, ROC curves, tables, and exported results | Project root |

Zenodo link:

```text
https://doi.org/10.5281/zenodo.20549960
```

After extracting the Zenodo ZIP, the local structure should look like this:

```text
repo/
|-- Data/
|-- Outputs/
|-- scripts/
|-- src/
|-- 1-EDA.ipynb
|-- 2-Data-Preprocessing-Time.ipynb
|-- 3-Data-Preprocessing-ML.ipynb
|-- 4-ML-Raw-Data-1-v4.ipynb
|-- 4-ML-Raw-Data-2-v4.ipynb
|-- 4-ML-Raw-Data-3-1-v4.ipynb
|-- 4-ML-Raw-Data-3-2-v4.ipynb
|-- README.md
`-- requirements.txt
```

`Data/`, `Outputs/`, `venv/`, and local temporary files are ignored by Git so the GitHub repository remains lightweight.

## Environment

Create the environment from the project root:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
```

Validate the installation, seed configuration, and available devices:

```powershell
python scripts\check_environment.py
```

Modern TensorFlow on native Windows does not expose CUDA GPUs. The main reproducible path in this repository is CPU-based, with fixed seeds and deterministic settings. For actual GPU training, use WSL2/Linux with a TensorFlow/CUDA-compatible stack.

## Reproducibility

Global seed:

```text
SEED = 42
```

The reproducibility setup is implemented in `src/reproducibility.py` and used by all notebooks. It covers:

- `PYTHONHASHSEED`
- `random.seed`
- `numpy.random.seed`
- `tf.keras.utils.set_random_seed`
- `TF_DETERMINISTIC_OPS`
- `TF_CUDNN_DETERMINISTIC`
- `TF_ENABLE_ONEDNN_OPTS=0`
- seeded Keras initializers
- seeded dropout layers
- `train_test_split(..., random_state=42)`

The notebooks are kept without embedded outputs so results can be regenerated from the data.

## Recommended Pipeline

The recommended reproduction path starts from the consolidated clean 20 s signals:

```text
Data/2-Consolidated-Time-Clean/emg_consolidated_time_clean.csv
```

The manual 20 s trimming step is already reflected in that file and does not need to be repeated for the main experiments.

| Step | Notebook | Purpose | Input | Output |
|---|---|---|---|---|
| 1 | `3-Data-Preprocessing-ML.ipynb` | Builds non-overlapping ML windows | `Data/2-Consolidated-Time-Clean/` | `Data/3-Data-ML-1/` |
| 2 | `4-ML-Raw-Data-1-v4.ipynb` | Trains and evaluates MLP overall 80/20 models | `Data/3-Data-ML-1/` | `Outputs/Results-Data-ML-1-v4/` |
| 3 | `4-ML-Raw-Data-2-v4.ipynb` | Trains and evaluates CNN1D within-user 80/20 models | `Data/3-Data-ML-1/` | `Outputs/Results-Data-ML-2-v4/` |
| 4 | `4-ML-Raw-Data-3-1-v4.ipynb` | Trains and evaluates MLP LOUO models | `Data/3-Data-ML-1/` | `Outputs/Results-Data-ML-3-1-v4-LOUO/` |
| 5 | `4-ML-Raw-Data-3-2-v4.ipynb` | Trains and evaluates CNN1D LOUO models | `Data/3-Data-ML-1/` | `Outputs/Results-Data-ML-3-2-v4-LOUO/` |

`1-EDA.ipynb` and `2-Data-Preprocessing-Time.ipynb` document the full path from `.emt` source files to clean time-domain signals. They are not required when starting from the Zenodo ZIP with clean consolidated signals.

## Time Windows

The clean signals are converted into non-overlapping windows:

| Window (s) | Samples/window | Features | Total rows | Rows/user |
|---:|---:|---:|---:|---:|
| 0.01 | 10 | 70 | 36,000 | 12,000 |
| 0.02 | 20 | 140 | 18,000 | 6,000 |
| 0.05 | 50 | 350 | 7,200 | 2,400 |
| 0.15 | 150 | 1,050 | 2,394 | 798 |
| 0.25 | 250 | 1,750 | 1,440 | 480 |
| 0.50 | 500 | 3,500 | 720 | 240 |
| 1.00 | 1000 | 7,000 | 360 | 120 |
| 2.00 | 2000 | 14,000 | 180 | 60 |

Each sample preserves sensor-wise temporal order: all points from `S1`, then `S2`, through `S7`.

## Models

MLP:

```text
Input -> Dense(1024) -> Dropout(0.3)
      -> Dense(512)  -> Dropout(0.3)
      -> Dense(128)  -> Dropout(0.2)
      -> Dense(6, softmax)
```

CNN1D:

```text
Flat input 7N -> reshape as time x sensors
Conv1D(32, k=9) -> MaxPool
Conv1D(64, k=5) -> MaxPool
Conv1D(128, k=3)
GlobalAveragePooling1D
Dense(128) -> Dropout(0.1)
Dense(64)  -> Dropout(0.1)
Dense(32)  -> Dropout(0.1)
Dense(6, softmax)
```

Common settings:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Loss | `sparse_categorical_crossentropy` |
| Batch size | 16 |
| Maximum epochs | 100 |
| Internal validation | 10% of training data |
| Early stopping | `val_loss`, patience 10 |
| Metrics | Accuracy, macro F1, macro AUC |

## Evaluation Protocols

| Protocol | Description | Purpose |
|---|---|---|
| Overall 80/20 | Pools the three users and splits train/test samples | Global separability when all users are represented |
| Within-user 80/20 | Splits each user independently | Main personalized calibration scenario |
| LOUO | Trains on two users and tests on the held-out third user | Exploratory transferability analysis |

The within-user protocol is the main personalized result. LOUO is reported as preliminary evidence of the difficulty of generalizing to unseen users.

## Execution Order

Open Jupyter from the project root:

```powershell
.\venv\Scripts\activate
jupyter notebook
```

Recommended order:

```text
3-Data-Preprocessing-ML.ipynb
4-ML-Raw-Data-1-v4.ipynb
4-ML-Raw-Data-2-v4.ipynb
4-ML-Raw-Data-3-1-v4.ipynb
4-ML-Raw-Data-3-2-v4.ipynb
```

To regenerate from `.emt` files first, run:

```text
1-EDA.ipynb
2-Data-Preprocessing-Time.ipynb
```

## Common Issues

| Issue | Fix |
|---|---|
| `FileNotFoundError` under `Data/...` | Confirm that the Zenodo ZIP was extracted at the project root and that `Data/` and `Outputs/` exist. |
| TensorFlow does not detect the GPU | This is expected on native Windows with TensorFlow >= 2.11. Use CPU or WSL2/Linux for CUDA GPU training. |
| Results differ unexpectedly | Start from a clean session, keep `SEED=42`, use the committed notebooks, and avoid changing major dependency versions. |
| A lowercase `outputs/` folder appears | Use the updated notebooks; the standard folder is `Outputs/`. |
| Jupyter opens in another folder | Start `jupyter notebook` from the project root. |

## Ethics and Use

The data are anonymized by participant code and are intended for academic and methodological use. The models are not diagnostic tools and do not replace clinical judgment.
