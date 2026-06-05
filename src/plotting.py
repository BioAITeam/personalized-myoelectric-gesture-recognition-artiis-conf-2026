from __future__ import annotations

import json
from pathlib import Path
from string import ascii_uppercase

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


USER_IDS = ("0007", "0008", "0010")


def save_png_pdf(fig, out_base: Path, dpi_png: int = 300) -> tuple[Path, Path]:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_base.with_name(f"{out_base.name}.png")
    pdf_path = out_base.with_name(f"{out_base.name}.pdf")
    fig.savefig(png_path, dpi=dpi_png, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    return png_path, pdf_path


def load_classes(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        values = json.load(fh)
    if isinstance(values, dict):
        values = values.get("classes", values)
    return [str(value) for value in values]


def load_vector(path: Path) -> np.ndarray:
    data = pd.read_csv(path)
    if data.empty:
        return np.array([], dtype=int)
    return data.iloc[:, 0].to_numpy()


def load_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy()


def draw_confusion_matrix(ax, cm: np.ndarray, class_names: list[str], title: str | None = None):
    cm = np.asarray(cm, dtype=int)
    n_classes = len(class_names)
    denom = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)
    cmap = mpl.cm.Blues
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    for row in range(n_classes):
        for col in range(n_classes):
            value = float(cm_pct[row, col])
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    facecolor=cmap(norm(value)),
                    edgecolor="white",
                    linewidth=0.7,
                )
            )
            color = "white" if value >= 0.55 else "#202020"
            ax.text(
                col,
                row,
                f"{value * 100:.1f}%\n({cm[row, col]})",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    ax.set_xlim(-0.5, n_classes - 0.5)
    ax.set_ylim(n_classes - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_confusion_from_arrays(y_true_int, y_pred_int, class_names, out_path: Path):
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true_int, y_pred_int, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_confusion_matrix(ax, cm, list(class_names))
    fig.tight_layout()
    paths = save_png_pdf(fig, out_path)
    plt.close(fig)
    return paths


def plot_history_axis(ax, history: pd.DataFrame, metric: str, title: str | None = None):
    train_col = metric
    val_col = f"val_{metric}"
    epoch = history["epoch"] if "epoch" in history.columns else np.arange(1, len(history) + 1)
    if train_col in history.columns:
        ax.plot(epoch, history[train_col], label="train", linewidth=1.8)
    if val_col in history.columns:
        ax.plot(epoch, history[val_col], label="validation", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)


def plot_training_curves(history: pd.DataFrame, out_base: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_history_axis(ax, history, "accuracy")
    fig.tight_layout()
    save_png_pdf(fig, Path(f"{out_base}_accuracy"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_history_axis(ax, history, "loss")
    fig.tight_layout()
    save_png_pdf(fig, Path(f"{out_base}_loss"))
    plt.close(fig)


def plot_roc_axis(ax, y_true_int: np.ndarray, y_prob: np.ndarray, class_names: list[str], title: str | None = None):
    y_true_int = np.asarray(y_true_int, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_classes = len(class_names)
    y_true_bin = np.eye(n_classes, dtype=int)[y_true_int]

    auc_values = []
    for idx, class_name in enumerate(class_names):
        if len(np.unique(y_true_bin[:, idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_prob[:, idx])
        auc_value = roc_auc_score(y_true_bin[:, idx], y_prob[:, idx])
        auc_values.append(auc_value)
        ax.plot(fpr, tpr, linewidth=1.4, label=f"{class_name} AUC={auc_value:.3f}")

    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if title:
        if auc_values:
            title = f"{title} | macro AUC={np.mean(auc_values):.3f}"
        ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def plot_roc_from_arrays(y_true_int, y_prob, class_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_roc_axis(ax, np.asarray(y_true_int), np.asarray(y_prob), list(class_names))
    fig.tight_layout()
    paths = save_png_pdf(fig, out_path)
    plt.close(fig)
    return paths


def _panel_label(ax, label: str):
    ax.text(
        -0.06,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _artifact_dir(results_dir: Path, stem: str, mode: str, user: str | None = None) -> Path:
    base = Path(results_dir) / stem
    if mode == "overall":
        return base / "overall"
    if mode == "by_user" and user:
        return base / "by_user" / f"user_{user}"
    if mode == "louo" and user:
        return base / "cross_user" / f"holdout_user_{user}"
    raise ValueError(f"Unsupported artifact request: mode={mode!r}, user={user!r}")


def _load_artifacts(result_dir: Path):
    arrays_dir = Path(result_dir) / "arrays"
    y_true = load_vector(arrays_dir / "y_true.csv").astype(int)
    y_pred = load_vector(arrays_dir / "y_pred.csv").astype(int)
    y_prob = load_matrix(arrays_dir / "y_prob.csv")
    classes_path = arrays_dir / "classes.json"
    if not classes_path.exists():
        classes_path = Path(result_dir) / "classes.json"
    classes = load_classes(classes_path)
    history = pd.read_csv(Path(result_dir) / "history.csv")
    return y_true, y_pred, y_prob, classes, history


def _available_items(items):
    return [item for item in items if item is not None]


def _write_labels_map(out_base: Path, items):
    map_path = Path(out_base).with_name(f"{Path(out_base).name}_labels_map.txt")
    with open(map_path, "w", encoding="utf-8") as fh:
        for idx, (title, _) in enumerate(items):
            fh.write(f"{ascii_uppercase[idx]}: {title}\n")


def _draw_collage(items, nrows: int, ncols: int, out_base: Path, kind: str):
    fig_w = 3.7 * ncols if kind == "confusion" else 4.1 * ncols
    fig_h = 3.7 * nrows if kind != "roc" else 3.5 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.ravel()

    for idx, ax in enumerate(axes_flat):
        if idx >= len(items):
            ax.axis("off")
            continue

        title, artifact_dir = items[idx]
        y_true, y_pred, y_prob, classes, history = _load_artifacts(artifact_dir)
        _panel_label(ax, ascii_uppercase[idx])
        if kind == "confusion":
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
            draw_confusion_matrix(ax, cm, classes, title=title)
        elif kind == "roc":
            plot_roc_axis(ax, y_true, y_prob, classes, title=title)
        elif kind == "accuracy":
            plot_history_axis(ax, history, "accuracy", title=title)
        elif kind == "loss":
            plot_history_axis(ax, history, "loss", title=title)
        else:
            raise ValueError(f"Unsupported collage kind: {kind!r}")

    fig.tight_layout()
    paths = save_png_pdf(fig, out_base)
    _write_labels_map(out_base, items)
    plt.close(fig)
    return paths


def build_overall_vector_collages(results_dir: Path, stems, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for stem in stems:
        artifact_dir = _artifact_dir(results_dir, stem, "overall")
        if (artifact_dir / "arrays" / "y_true.csv").exists():
            items.append((stem.replace("ml_win_", "").replace("s", " s"), artifact_dir))

    items = _available_items(items)
    if not items:
        return {}

    outputs = {}
    outputs["confusion"] = _draw_collage(items, 2, 4, out_dir / "overall_confusion_2x4", "confusion")
    outputs["roc"] = _draw_collage(items, 2, 4, out_dir / "overall_roc_2x4", "roc")
    outputs["accuracy"] = _draw_collage(items, 2, 4, out_dir / "overall_accuracy_2x4", "accuracy")
    outputs["loss"] = _draw_collage(items, 2, 4, out_dir / "overall_loss_2x4", "loss")
    return outputs


def build_by_user_vector_collages_for_stem(results_dir: Path, stem: str, users=USER_IDS):
    coll_dir = Path(results_dir) / stem / "by_user" / "_collages"
    coll_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for user in users:
        artifact_dir = _artifact_dir(results_dir, stem, "by_user", user)
        if (artifact_dir / "arrays" / "y_true.csv").exists():
            items.append((f"user {user}", artifact_dir))

    if not items:
        return {}

    outputs = {}
    outputs["confusion"] = _draw_collage(items, 1, 3, coll_dir / f"{stem}_byuser_confusion_1x3", "confusion")
    outputs["roc"] = _draw_collage(items, 1, 3, coll_dir / f"{stem}_byuser_roc_1x3", "roc")
    outputs["accuracy"] = _draw_collage(items, 1, 3, coll_dir / f"{stem}_byuser_accuracy_1x3", "accuracy")
    outputs["loss"] = _draw_collage(items, 1, 3, coll_dir / f"{stem}_byuser_loss_1x3", "loss")
    return outputs


def build_louo_vector_collages_for_stem(results_dir: Path, stem: str, users=USER_IDS):
    coll_dir = Path(results_dir) / stem / "cross_user" / "_collages"
    coll_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for user in users:
        artifact_dir = _artifact_dir(results_dir, stem, "louo", user)
        if (artifact_dir / "arrays" / "y_true.csv").exists():
            items.append((f"held-out {user}", artifact_dir))

    if not items:
        return {}

    outputs = {}
    outputs["confusion"] = _draw_collage(items, 1, 3, coll_dir / f"{stem}_louo_confusion_1x3", "confusion")
    outputs["roc"] = _draw_collage(items, 1, 3, coll_dir / f"{stem}_louo_roc_1x3", "roc")
    outputs["accuracy"] = _draw_collage(items, 1, 3, coll_dir / f"{stem}_louo_accuracy_1x3", "accuracy")
    outputs["loss"] = _draw_collage(items, 1, 3, coll_dir / f"{stem}_louo_loss_1x3", "loss")
    return outputs
