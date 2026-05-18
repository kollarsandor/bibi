import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

APP_NAME = "jaide-v40-training"
GPU_SPEC = "B200:8"
DATA_VOLUME_NAME = "jaide-training-data"
CHECKPOINT_VOLUME_NAME = "jaide-checkpoints"

DATA_MOUNT_PATH = Path("/data")
CHECKPOINT_MOUNT_PATH = Path("/checkpoints")
PROJECT_MOUNT_PATH = Path("/jaide")

DATASET_DIR = DATA_MOUNT_PATH / "dataset"
DATASET_FILE = DATASET_DIR / "train.jsonl"
DATASET_METADATA_FILE = DATASET_DIR / "metadata.json"

MODELS_DIR = PROJECT_MOUNT_PATH / "models"
BINARY_PATH = PROJECT_MOUNT_PATH / "main"

CPU_REQUEST = 64.0
CPU_LIMIT = 80.0
MEMORY_REQUEST_MB = 262144
MEMORY_LIMIT_MB = 262144
EPHEMERAL_DISK_MB = 3145728
TIMEOUT_SECONDS = 86400

LOCAL_PROJECT_DIR = (Path(__file__).resolve().parent / "../..").resolve()

IGNORE_PATTERNS = [
    "node_modules",
    ".git",
    "zig-cache",
    ".pythonlibs",
    ".cache",
    ".upm",
    "__pycache__",
    ".local",
    ".replit",
    "*.bin",
]

app = modal.App(APP_NAME)

jaide_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .run_commands(
        "DEBIAN_FRONTEND=noninteractive apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-change-held-packages git curl xz-utils build-essential wget ca-certificates",
        "rm -rf /var/lib/apt/lists/*",
    )
    .pip_install("pyarrow", "requests", "zstandard", "datasets", "huggingface_hub", "hf_xet")
    .run_commands(
        "mkdir -p /opt",
        "curl -sL https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz | tar -xJ -C /opt",
        "ln -sf /opt/zig-linux-x86_64-0.13.0/zig /usr/local/bin/zig",
        "zig version",
    )
    .env(
        {
            "PATH": "/opt/zig-linux-x86_64-0.13.0:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HF_HOME": "/data/hf_home",
            "HF_DATASETS_CACHE": "/data/hf_datasets_cache",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_dir(
        str(LOCAL_PROJECT_DIR),
        remote_path=str(PROJECT_MOUNT_PATH),
        ignore=IGNORE_PATTERNS,
    )
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)


def _run_checked(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        return p.returncode, p.stdout or "", p.stderr or ""
    except FileNotFoundError as e:
        return 127, "", str(e)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
        if isinstance(value, dict):
            return value
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _write_json_file(path: Path, value: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _count_lines(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _extract_text_from_row(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("text", "content", "sentence", "article"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    for val in row.values():
        if isinstance(val, str) and len(val.strip()) > 50:
            return val.strip()
    return ""


def download_finephrase_to_jsonl(volume: modal.Volume) -> Tuple[str, int, int]:
    from datasets import load_dataset

    _ensure_dir(DATASET_DIR)

    if DATASET_FILE.is_file() and DATASET_FILE.stat().st_size > 0:
        size = int(DATASET_FILE.stat().st_size)
        metadata = _read_json_file(DATASET_METADATA_FILE)
        if metadata is not None and int(metadata.get("dataset_size", -1)) == size and int(metadata.get("line_count", 0)) > 0:
            return str(DATASET_FILE), size, int(metadata["line_count"])
        line_count = _count_lines(DATASET_FILE)
        if line_count > 0:
            _write_json_file(DATASET_METADATA_FILE, {"dataset_path": str(DATASET_FILE), "dataset_size": size, "line_count": line_count})
            volume.commit()
            return str(DATASET_FILE), size, line_count
        DATASET_FILE.unlink()

    tmp_file = DATASET_FILE.with_suffix(".jsonl.tmp")
    if tmp_file.exists():
        tmp_file.unlink()

    ds = load_dataset("HuggingFaceFW/finephrase", split="train")

    line_count = 0
    with open(tmp_file, "w", encoding="utf-8") as f_out:
        for row in ds:
            text = _extract_text_from_row(row)
            if text and len(text) > 20:
                f_out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                line_count += 1

    if line_count <= 0:
        if tmp_file.exists():
            tmp_file.unlink()
        raise RuntimeError("Dataset conversion produced zero usable samples")

    tmp_file.replace(DATASET_FILE)
    size = int(DATASET_FILE.stat().st_size)
    _write_json_file(DATASET_METADATA_FILE, {"dataset_path": str(DATASET_FILE), "dataset_size": size, "line_count": line_count})
    volume.commit()
    return str(DATASET_FILE), size, line_count


def _build_zig(project_dir: str) -> None:
    if BINARY_PATH.is_file():
        return
    rc, out, err = _run_checked(
        ["zig", "build-exe", "src/main.zig", "-O", "ReleaseFast", f"-femit-bin={BINARY_PATH}"],
        cwd=project_dir,
    )
    if rc != 0:
        raise RuntimeError(f"Build failed with exit code {rc}: {(err or out)[-8000:]}")
    if not BINARY_PATH.is_file():
        raise FileNotFoundError(f"Built binary not found at {BINARY_PATH}")
    BINARY_PATH.chmod(0o755)


def _detect_gpus() -> Tuple[int, str]:
    try:
        p = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True)
    except FileNotFoundError as e:
        return 0, str(e)
    output = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    lines = [l for l in (p.stdout or "").splitlines() if l.strip()]
    return len(lines), output


def _expected_gpu_count() -> int:
    if ":" not in GPU_SPEC:
        return 1
    try:
        return int(GPU_SPEC.rsplit(":", 1)[1])
    except ValueError:
        return 1


def _extract_loss(stdout: str) -> Optional[float]:
    if not stdout:
        return None
    loss_value = None
    pattern = re.compile(r"\b(?:loss|train_loss|training_loss)\b[^0-9+\-]*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)", re.IGNORECASE)
    for line in stdout.splitlines():
        match = pattern.search(line)
        if match:
            try:
                loss_value = float(match.group(1))
            except ValueError:
                continue
    return loss_value


def _read_tail(path: Path, max_chars: int = 8000) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    byte_count = min(size, max_chars * 4)
    with open(path, "rb") as f:
        if size > byte_count:
            f.seek(size - byte_count)
        data = f.read()
    return data.decode("utf-8", errors="replace")[-max_chars:]


def _run_training_command(cmd: List[str], env: Dict[str, str], timeout_seconds: int, epoch: int) -> Tuple[int, str, str, bool]:
    logs_dir = Path("/tmp/jaide_training_logs")
    _ensure_dir(logs_dir)
    stdout_path = logs_dir / f"epoch_{epoch:03d}.stdout.log"
    stderr_path = logs_dir / f"epoch_{epoch:03d}.stderr.log"

    timed_out = False
    with open(stdout_path, "w", encoding="utf-8", errors="replace") as stdout_file, open(stderr_path, "w", encoding="utf-8", errors="replace") as stderr_file:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                env=env,
                cwd=str(PROJECT_MOUNT_PATH),
                preexec_fn=os.setsid,
            )
            try:
                return_code = proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, 15)
                    proc.wait(timeout=30)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(proc.pid, 9)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                return_code = -9
        except FileNotFoundError as e:
            stderr_file.write(str(e))
            return_code = 127

    stdout_tail = _read_tail(stdout_path)
    stderr_tail = _read_tail(stderr_path)
    return int(return_code), stdout_tail, stderr_tail, timed_out


def _validate_positive_int(value: int, name: str) -> int:
    converted = int(value)
    if converted < 1:
        raise ValueError(f"{name} must be a positive integer")
    return converted


@app.function(
    image=jaide_image,
    gpu=GPU_SPEC,
    cpu=(CPU_REQUEST, CPU_LIMIT),
    memory=(MEMORY_REQUEST_MB, MEMORY_LIMIT_MB),
    ephemeral_disk=EPHEMERAL_DISK_MB,
    timeout=TIMEOUT_SECONDS,
    volumes={
        str(DATA_MOUNT_PATH): data_volume,
        str(CHECKPOINT_MOUNT_PATH): checkpoint_volume,
    },
)
def train_jaide(
    epochs: int = 20,
    batch_size: int = 256,
    per_epoch_timeout_seconds: int = 3600,
) -> Dict[str, Any]:
    num_epochs = _validate_positive_int(epochs, "epochs")
    bs = _validate_positive_int(batch_size, "batch_size")
    epoch_timeout = _validate_positive_int(per_epoch_timeout_seconds, "per_epoch_timeout_seconds")

    data_volume.reload()
    checkpoint_volume.reload()

    gpu_count, gpu_list = _detect_gpus()
    expected_gpus = _expected_gpu_count()
    if gpu_count < 1:
        raise RuntimeError(f"No NVIDIA GPUs detected: {gpu_list}")
    if gpu_count != expected_gpus:
        raise RuntimeError(f"Expected {expected_gpus} GPUs from {GPU_SPEC}, detected {gpu_count}: {gpu_list}")

    dataset_path, dataset_size, sample_count = download_finephrase_to_jsonl(data_volume)
    if sample_count <= 0:
        raise RuntimeError("Dataset contains zero samples")

    os.chdir(str(PROJECT_MOUNT_PATH))
    _ensure_dir(MODELS_DIR)
    _ensure_dir(CHECKPOINT_MOUNT_PATH)

    _build_zig(str(PROJECT_MOUNT_PATH))

    total_start = time.time()
    training_results: List[Dict[str, Any]] = []
    aborted = False

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))

        cmd = [
            str(BINARY_PATH),
            "--mode", "train",
            "--epochs", "1",
            "--batch-size", str(bs),
            "--dataset-path", str(dataset_path),
            "--samples", str(sample_count),
            "--output-dir", str(MODELS_DIR),
        ]

        return_code, stdout_tail, stderr_tail, timed_out = _run_training_command(cmd, env, epoch_timeout, epoch)

        epoch_time = float(time.time() - epoch_start)
        loss = _extract_loss(stdout_tail)
        if loss is None:
            loss = 0.0

        checkpoint_dir = CHECKPOINT_MOUNT_PATH / f"epoch_{epoch:03d}"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        _ensure_dir(checkpoint_dir)

        model_files = sorted(MODELS_DIR.glob("*.bin"))
        for model_file in model_files:
            shutil.copy2(model_file, checkpoint_dir / model_file.name)

        ckpt_files = list(checkpoint_dir.glob("*.bin"))
        ckpt_size = int(sum(f.stat().st_size for f in ckpt_files)) if ckpt_files else 0

        _write_json_file(
            checkpoint_dir / "metadata.json",
            {
                "epoch": epoch,
                "loss": loss,
                "duration_seconds": epoch_time,
                "batch_size": bs,
                "dataset_samples": sample_count,
                "dataset_size_mb": float(dataset_size) / 1e6,
                "gpu_config": f"{gpu_count}x NVIDIA B200",
                "return_code": int(return_code),
                "timed_out": bool(timed_out),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "checkpoint_bytes": ckpt_size,
                "checkpoint_files": len(ckpt_files),
            },
        )

        checkpoint_volume.commit()

        training_results.append(
            {
                "epoch": epoch,
                "loss": loss,
                "time_seconds": epoch_time,
                "checkpoint_mb": float(ckpt_size) / 1e6,
                "files": len(ckpt_files),
                "return_code": int(return_code),
                "timed_out": bool(timed_out),
            }
        )

        if return_code != 0 or timed_out:
            aborted = True
            break

    total_time = float(time.time() - total_start)

    successful_results = [r for r in training_results if int(r["return_code"]) == 0 and not r.get("timed_out", False)]

    latest_dir = CHECKPOINT_MOUNT_PATH / "latest"
    if latest_dir.is_symlink():
        latest_dir.unlink()
    elif latest_dir.exists():
        try:
            shutil.rmtree(latest_dir)
        except OSError:
            pass

    if successful_results:
        last_ok_epoch = int(successful_results[-1]["epoch"])
        src = CHECKPOINT_MOUNT_PATH / f"epoch_{last_ok_epoch:03d}"
        if src.is_dir():
            shutil.copytree(src, latest_dir)

    final_loss = float(successful_results[-1]["loss"]) if successful_results else 0.0
    completed_epochs = len(successful_results)
    status = "completed" if (not aborted and completed_epochs == num_epochs) else "failed"

    _write_json_file(
        CHECKPOINT_MOUNT_PATH / "training_complete.json",
        {
            "status": status,
            "total_epochs": num_epochs,
            "completed_epochs": completed_epochs,
            "attempted_epochs": len(training_results),
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60.0,
            "final_loss": final_loss,
            "dataset_path": str(dataset_path),
            "dataset_size_mb": float(dataset_size) / 1e6,
            "sample_count": sample_count,
            "gpu_config": f"{gpu_count}x NVIDIA B200",
            "gpu_list": gpu_list,
            "epochs": training_results,
        },
    )

    checkpoint_volume.commit()

    return {
        "status": status,
        "epochs": num_epochs,
        "completed_epochs": completed_epochs,
        "final_loss": final_loss,
        "total_time_minutes": total_time / 60.0,
        "dataset_size_mb": float(dataset_size) / 1e6,
        "sample_count": sample_count,
        "gpu_config": f"{gpu_count}x NVIDIA B200",
        "checkpoints": completed_epochs,
    }


@app.local_entrypoint()
def main(
    epochs: int = 20,
    batch_size: int = 256,
    per_epoch_timeout_seconds: int = 3600,
) -> None:
    result = train_jaide.remote(
        epochs=epochs,
        batch_size=batch_size,
        per_epoch_timeout_seconds=per_epoch_timeout_seconds,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
