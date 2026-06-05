from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "reproducibility_audit"
LOG_DIR = AUDIT_DIR / "logs"
SNAPSHOT_DIR = AUDIT_DIR / "snapshots"

NOTEBOOKS = [
    "1-EDA.ipynb",
    "2-Data-Preprocessing-Time.ipynb",
    "3-Data-Preprocessing-ML.ipynb",
    "4-ML-Raw-Data-1-v4.ipynb",
    "4-ML-Raw-Data-2-v4.ipynb",
    "4-ML-Raw-Data-3-1-v4.ipynb",
    "4-ML-Raw-Data-3-2-v4.ipynb",
]

SCOPES = {
    "1-EDA.ipynb": [
        REPO_ROOT / "Outputs" / "summary.csv",
        REPO_ROOT / "Data" / "2-Consolidated",
    ],
    "2-Data-Preprocessing-Time.ipynb": [
        REPO_ROOT / "Data" / "2-Consolidated-Time-Clean",
    ],
    "3-Data-Preprocessing-ML.ipynb": [
        REPO_ROOT / "Data" / "3-Data-ML-1",
    ],
    "4-ML-Raw-Data-1-v4.ipynb": [
        REPO_ROOT / "Outputs" / "Results-Data-ML-1-v4",
    ],
    "4-ML-Raw-Data-2-v4.ipynb": [
        REPO_ROOT / "Outputs" / "Results-Data-ML-2-v4",
    ],
    "4-ML-Raw-Data-3-1-v4.ipynb": [
        REPO_ROOT / "Outputs" / "Results-Data-ML-3-1-v4-LOUO",
    ],
    "4-ML-Raw-Data-3-2-v4.ipynb": [
        REPO_ROOT / "Outputs" / "Results-Data-ML-3-2-v4-LOUO",
    ],
}

INCLUDED_SUFFIXES = {".csv", ".json", ".txt", ".tex"}
EXCLUDED_NAMES = {"metrics.txt", "info.txt"}
TIME_KEYS = {"train_time_s", "test_time_s"}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def write_status(data: dict) -> None:
    ensure_dirs()
    (AUDIT_DIR / "status.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_audit_log(message: str) -> None:
    ensure_dirs()
    with open(AUDIT_DIR / "audit.log", "a", encoding="utf-8") as fh:
        fh.write(f"[{now()}] {message}\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_json(value):
    if isinstance(value, dict):
        return {
            key: normalize_json(item)
            for key, item in sorted(value.items())
            if key not in TIME_KEYS
        }
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    return value


def file_entry(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            normalized = normalize_json(value)
            payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")
            return {"kind": "json_normalized", "sha256": sha256_bytes(payload)}
        except Exception:
            pass
    return {"kind": "bytes", "sha256": sha256_bytes(path.read_bytes())}


def iter_scope_files(paths: list[Path]):
    for path in paths:
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(item for item in path.rglob("*") if item.is_file())
        else:
            candidates = []

        for candidate in candidates:
            if candidate.suffix.lower() not in INCLUDED_SUFFIXES:
                continue
            if candidate.name in EXCLUDED_NAMES:
                continue
            yield candidate


def snapshot_notebook_outputs(notebook_name: str, run_index: int) -> dict:
    paths = SCOPES[notebook_name]
    entries = {}
    for file_path in iter_scope_files(paths):
        rel = file_path.relative_to(REPO_ROOT).as_posix()
        entries[rel] = file_entry(file_path)

    snapshot = {
        "run": run_index,
        "notebook": notebook_name,
        "created_at": now(),
        "file_count": len(entries),
        "files": entries,
    }

    run_dir = SNAPSHOT_DIR / f"run_{run_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{Path(notebook_name).stem}.json"
    out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    return snapshot


def extract_metric_rows(run_index: int) -> list[dict]:
    rows = []
    for metrics_path in sorted((REPO_ROOT / "Outputs").glob("Results-Data-ML-*/**/metrics.json")):
        try:
            raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = raw.get("metrics", {})
        timings = raw.get("timings", {})
        row = {
            "run": run_index,
            "path": metrics_path.relative_to(REPO_ROOT).as_posix(),
            "experiment": raw.get("experiment", ""),
            "stem": raw.get("stem", ""),
            "tag": raw.get("tag", ""),
            "epochs_trained": timings.get("epochs_trained", ""),
        }
        for key, value in sorted(metrics.items()):
            row[key] = value
        rows.append(row)
    return rows


def write_metric_rows(rows: list[dict]) -> None:
    if not rows:
        return
    all_keys = []
    for row in rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)
    out_path = AUDIT_DIR / "metrics_by_run.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def output_to_text(output: dict) -> str:
    output_type = output.get("output_type")
    if output_type == "stream":
        text = output.get("text", "")
        if isinstance(text, list):
            return "".join(text)
        return text
    if output_type == "error":
        traceback = output.get("traceback", [])
        return "\n".join(traceback) + "\n"
    data = output.get("data", {})
    if "text/plain" in data:
        text = data["text/plain"]
        if isinstance(text, list):
            return "".join(text)
        return str(text) + "\n"
    return ""


def execute_notebook(notebook_name: str, run_index: int) -> float:
    notebook_path = REPO_ROOT / notebook_name
    log_path = LOG_DIR / f"run_{run_index:02d}_{Path(notebook_name).stem}.log"
    append_audit_log(f"START run={run_index} notebook={notebook_name}")
    start = time.perf_counter()

    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=-1,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )

    try:
        client.execute()
    finally:
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(f"run={run_index}\nnotebook={notebook_name}\n")
            fh.write(f"started_at={now()}\n\n")
            for index, cell in enumerate(nb.cells, start=1):
                if cell.get("cell_type") != "code":
                    continue
                outputs = cell.get("outputs", [])
                if not outputs:
                    continue
                fh.write(f"\n--- cell {index} ---\n")
                for output in outputs:
                    fh.write(output_to_text(output))

    elapsed = time.perf_counter() - start
    append_audit_log(f"END run={run_index} notebook={notebook_name} elapsed_s={elapsed:.3f}")
    return elapsed


def load_snapshot(run_index: int, notebook_name: str) -> dict:
    path = SNAPSHOT_DIR / f"run_{run_index:02d}" / f"{Path(notebook_name).stem}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def compare_snapshots(runs: int) -> list[dict]:
    rows = []
    for notebook_name in NOTEBOOKS:
        baseline = load_snapshot(1, notebook_name).get("files", {})
        for run_index in range(2, runs + 1):
            current = load_snapshot(run_index, notebook_name).get("files", {})
            missing = sorted(set(baseline) - set(current))
            extra = sorted(set(current) - set(baseline))
            changed = sorted(
                key for key in set(baseline).intersection(current)
                if baseline[key]["sha256"] != current[key]["sha256"]
            )
            rows.append(
                {
                    "notebook": notebook_name,
                    "baseline_run": 1,
                    "compared_run": run_index,
                    "baseline_files": len(baseline),
                    "current_files": len(current),
                    "missing": len(missing),
                    "extra": len(extra),
                    "changed": len(changed),
                    "first_missing": missing[0] if missing else "",
                    "first_extra": extra[0] if extra else "",
                    "first_changed": changed[0] if changed else "",
                }
            )
    return rows


def write_comparison(rows: list[dict]) -> None:
    out_csv = AUDIT_DIR / "comparison_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (AUDIT_DIR / "comparison_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_audit(runs: int) -> None:
    ensure_dirs()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONHASHSEED", "42")
    metric_rows = []
    timings = []

    for run_index in range(1, runs + 1):
        for position, notebook_name in enumerate(NOTEBOOKS, start=1):
            write_status(
                {
                    "state": "running",
                    "run": run_index,
                    "runs": runs,
                    "notebook_index": position,
                    "notebook_count": len(NOTEBOOKS),
                    "notebook": notebook_name,
                    "updated_at": now(),
                }
            )
            elapsed = execute_notebook(notebook_name, run_index)
            snapshot = snapshot_notebook_outputs(notebook_name, run_index)
            timings.append(
                {
                    "run": run_index,
                    "notebook": notebook_name,
                    "elapsed_s": round(elapsed, 3),
                    "file_count": snapshot["file_count"],
                }
            )
        metric_rows.extend(extract_metric_rows(run_index))
        write_metric_rows(metric_rows)

    comparison_rows = compare_snapshots(runs)
    write_comparison(comparison_rows)
    with open(AUDIT_DIR / "execution_timings.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run", "notebook", "elapsed_s", "file_count"])
        writer.writeheader()
        writer.writerows(timings)

    total_changed = sum(row["changed"] + row["missing"] + row["extra"] for row in comparison_rows)
    write_status(
        {
            "state": "complete",
            "runs": runs,
            "total_differences": total_changed,
            "updated_at": now(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    run_audit(args.runs)


if __name__ == "__main__":
    main()
