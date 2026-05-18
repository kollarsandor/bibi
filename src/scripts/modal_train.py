import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import modal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = modal.App("jaide-v40-training")

volume = modal.Volume.from_name("jaide-training-data", create_if_missing=True)

LOCAL_SRC_DIR = (Path(__file__).resolve().parent / "..").resolve()

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        "DEBIAN_FRONTEND=noninteractive apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-change-held-packages "
        "build-essential wget curl git ca-certificates xz-utils libgomp1",
        "rm -rf /var/lib/apt/lists/*",
    )
    .pip_install("datasets", "huggingface_hub")
    .run_commands(
        # Install Zig 0.13.0
        "curl -sSf https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz -o /tmp/zig.tar.xz",
        "tar -xf /tmp/zig.tar.xz -C /tmp",
        "mv /tmp/zig-linux-x86_64-0.13.0 /usr/local/zig",
        "ln -sf /usr/local/zig/zig /usr/local/bin/zig",
        "zig version",
        # Install Futhark pre-built binary (nightly, much faster than opam compile)
        "curl -sSfL https://github.com/diku-dk/futhark/releases/download/nightly/futhark-nightly-linux-x86_64.tar.xz -o /tmp/futhark.tar.xz",
        "tar -xf /tmp/futhark.tar.xz -C /tmp",
        "cp /tmp/futhark-nightly-linux-x86_64/bin/futhark /usr/local/bin/futhark",
        "chmod +x /usr/local/bin/futhark",
        "futhark --version",
    )
    .add_local_dir(
        str(LOCAL_SRC_DIR),
        remote_path="/jaide_src",
        copy=True,
    )
    .run_commands(
        # Sync Futhark package dependencies (diku-dk/sorts etc.) before compiling
        "cd /jaide_src/hw/accel && futhark pkg sync 2>&1 | tail -10 "
        "|| echo 'futhark pkg sync failed, will retry at runtime'",
        # Compile futhark kernels to C library (--library generates .c + .h for linking)
        "futhark c --library /jaide_src/hw/accel/futhark_kernels.fut "
        "-o /jaide_src/hw/accel/futhark_kernels 2>&1 | tail -10 "
        "|| echo 'Futhark compile will retry at runtime'",
        "ls /jaide_src/hw/accel/futhark_kernels.c /jaide_src/hw/accel/futhark_kernels.h 2>/dev/null && echo 'Futhark C OK' || echo 'Will retry at runtime'",
        # Build JAIDE via build.zig (Zig 0.13.0 compatible — no --main-pkg-path)
        "cd /jaide_src && zig build -Doptimize=ReleaseFast 2>&1 | tail -30 "
        "|| echo 'Zig build will retry at runtime'",
        "ls /jaide_src/zig-out/bin/jaide 2>/dev/null && echo 'Binary ready' || echo 'Binary will build at runtime'",
    )
)

WORK_DIR = Path("/workspace")
DATASET_PATH = Path("/dataset/train.jsonl")
MODELS_DIR = Path("/models")
BINARY_PATH = Path("/jaide_src/zig-out/bin/jaide")

GPU_COUNT = 8
GPU_TYPE = "B200"
DEFAULT_GPU_CONFIG = f"{GPU_COUNT}x {GPU_TYPE}"


class TrainingParameters(TypedDict):
    epochs: int
    batch_size: int
    learning_rate: float
    dim: int
    layers: int
    sample_limit: int
    noise_level: float
    gradient_clip: float


class TrainingResult(TypedDict):
    status: str
    exit_code: int
    duration_seconds: float
    gpu_config: str
    model_path: Optional[str]
    stdout: str
    stderr: str
    timestamp: float
    parameters: Optional[TrainingParameters]


class ModelInfo(TypedDict):
    filename: str
    size_bytes: int
    size_mb: float
    modified: float
    path: str


class ListModelsResult(TypedDict):
    models: List[ModelInfo]
    training_logs: List[TrainingResult]


def _download_finephrase_to_jsonl() -> None:
    from datasets import load_dataset

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.is_file() and DATASET_PATH.stat().st_size > 0:
        logger.info("Dataset already present, skipping download.")
        return
    logger.info("Downloading HuggingFaceFW/finephrase dataset...")
    ds = load_dataset("HuggingFaceFW/finephrase", split="train")
    with open(DATASET_PATH, "w", encoding="utf-8") as f_out:
        for row in ds:
            text = ""
            for key in ("text", "content", "sentence", "article"):
                val = row.get(key) if isinstance(row, dict) else None
                if isinstance(val, str) and val.strip():
                    text = val.strip()
                    break
            if not text and isinstance(row, dict):
                for _, val in row.items():
                    if isinstance(val, str) and len(val) > 50:
                        text = val.strip()
                        break
            if text and len(text) > 20:
                f_out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    logger.info(f"Dataset saved to {DATASET_PATH}")


def _ensure_positive(name: str, value: Union[int, float]) -> float:
    val = float(value)
    if val <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return val


def _validate_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got bool")
    if not isinstance(value, int):
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value}")
    return value


def _runtime_build() -> None:
    """Fallback: compile binary at runtime if image build did not produce it."""
    if BINARY_PATH.is_file():
        logger.info("Pre-built binary found.")
        return
    logger.info("Binary not found — attempting runtime build...")
    src = Path("/jaide_src")
    accel_dir = src / "hw/accel"
    futhark_c = accel_dir / "futhark_kernels.c"

    # Step 1: sync futhark packages
    if not (accel_dir / "lib").is_dir():
        logger.info("Running futhark pkg sync...")
        r = subprocess.run(
            ["futhark", "pkg", "sync"],
            capture_output=True, text=True, cwd=str(accel_dir),
        )
        if r.returncode != 0:
            logger.warning(f"futhark pkg sync warning: {r.stderr[:500]}")

    # Step 2: compile .fut -> .c + .h (--library mode; always regenerate)
    logger.info("Generating futhark_kernels.c/.h from .fut source (--library)...")
    r = subprocess.run(
        ["futhark", "c", "--library",
         str(accel_dir / "futhark_kernels.fut"),
         "-o", str(accel_dir / "futhark_kernels")],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        logger.warning(f"Futhark compile warning: {r.stderr[:500]}")
    else:
        logger.info("Futhark library compiled OK.")

    # Step 3: build via build.zig (Zig 0.13.0 compatible)
    r = subprocess.run(
        ["zig", "build", "-Doptimize=ReleaseFast"],
        capture_output=True, text=True, cwd=str(src),
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"Runtime build failed:\nSTDOUT:\n{r.stdout[:4000]}\nSTDERR:\n{r.stderr[:4000]}"
        )
    logger.info("Runtime build succeeded.")


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    timeout=86400,
    volumes={
        str(MODELS_DIR): volume,
        str(DATASET_PATH.parent): modal.Volume.from_name("jaide-dataset", create_if_missing=True),
    },
    cpu=64.0,
    memory=128 * 1024,
)
def train_jaide_rsf(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dim: int,
    layers: int,
    sample_limit: int,
    noise_level: float,
    gradient_clip: float,
) -> TrainingResult:
    start_wall = time.time()

    try:
        epochs = _validate_int("epochs", epochs)
        batch_size = _validate_int("batch_size", batch_size)
        dim = _validate_int("dim", dim)
        layers = _validate_int("layers", layers)
        sample_limit = _validate_int("sample_limit", sample_limit)

        learning_rate = float(_ensure_positive("learning_rate", learning_rate))
        noise_level = float(max(0.0, float(noise_level)))
        gradient_clip = float(max(0.0, float(gradient_clip)))

        WORK_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        _runtime_build()

        if not DATASET_PATH.is_file():
            _download_finephrase_to_jsonl()

        if not DATASET_PATH.is_file():
            raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

        unique_id = uuid.uuid4().hex
        model_output = MODELS_DIR / f"jaide30b_{GPU_COUNT}x_{unique_id}.bin"

        # main.zig CLI flags (verified against parseArgs in main.zig)
        train_args = [
            str(BINARY_PATH),
            "--mode", "train",
            "--dataset-path", str(DATASET_PATH),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(learning_rate),          # --lr, not --learning-rate
            "--embedding-dim", str(dim),
            "--layers", str(layers),
            "--sample-limit", str(sample_limit),
            "--gradient-clip", str(gradient_clip),
            "--models-dir", str(MODELS_DIR),     # saves to {MODELS_DIR}/rsf_trained.bin
        ]

        env = os.environ.copy()
        env.update({
            "JAIDE_GPU_COUNT": str(GPU_COUNT),
            "JAIDE_GPU_TYPE": GPU_TYPE,
            "NCCL_DEBUG": "INFO",
            "NCCL_IB_DISABLE": "0",
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(GPU_COUNT)),
        })

        out_log = WORK_DIR / "train_stdout.log"
        err_log = WORK_DIR / "train_stderr.log"

        train_start = time.time()
        with open(out_log, "w", encoding="utf-8") as tout, \
             open(err_log, "w", encoding="utf-8") as terr:
            proc = subprocess.run(train_args, stdout=tout, stderr=terr, env=env, cwd=str(WORK_DIR))
        train_end = time.time()

        stdout_txt = out_log.read_text(encoding="utf-8", errors="replace")
        stderr_txt = err_log.read_text(encoding="utf-8", errors="replace")

        status = "completed" if proc.returncode == 0 else "failed"
        model_path_str = str(model_output) if proc.returncode == 0 and model_output.is_file() else None

        result: TrainingResult = {
            "status": status,
            "exit_code": proc.returncode,
            "duration_seconds": train_end - train_start,
            "gpu_config": DEFAULT_GPU_CONFIG,
            "model_path": model_path_str,
            "stdout": stdout_txt,
            "stderr": stderr_txt,
            "timestamp": train_end,
            "parameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "dim": dim,
                "layers": layers,
                "sample_limit": sample_limit,
                "noise_level": noise_level,
                "gradient_clip": gradient_clip,
            },
        }

        log_path = MODELS_DIR / f"training_log_{int(train_end)}_{unique_id}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        volume.commit()
        return result

    except Exception as e:
        end_wall = time.time()
        logger.exception("Training failed with exception")
        return {
            "status": "exception",
            "exit_code": 1,
            "duration_seconds": end_wall - start_wall,
            "gpu_config": DEFAULT_GPU_CONFIG,
            "model_path": None,
            "stdout": "",
            "stderr": f"{type(e).__name__}: {e}",
            "timestamp": end_wall,
            "parameters": None,
        }


@app.function(image=image, volumes={str(MODELS_DIR): volume})
def list_models() -> ListModelsResult:
    volume.reload()
    models: List[ModelInfo] = []
    try:
        for f in MODELS_DIR.glob("jaide30b_*.bin"):
            stat = f.stat()
            models.append({
                "filename": f.name,
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 * 1024),
                "modified": stat.st_mtime,
                "path": str(f),
            })
    except OSError as e:
        logger.error(f"Failed to list models: {e}")

    logs: List[Any] = []
    try:
        for f in MODELS_DIR.glob("training_log_*.json"):
            try:
                with open(f, "r", encoding="utf-8") as lf:
                    data = json.load(lf)
                if isinstance(data, dict):
                    logs.append(data)
            except (json.JSONDecodeError, TypeError, ValueError) as ex:
                logger.warning(f"Skipping malformed log {f.name}: {ex}")
    except OSError as e:
        logger.error(f"Failed to scan logs: {e}")

    return {
        "models": sorted(models, key=lambda x: float(x.get("modified", 0.0)), reverse=True),
        "training_logs": sorted(logs, key=lambda x: float(x.get("timestamp", 0.0)), reverse=True),
    }


@app.local_entrypoint()
def main(
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.0005,
    dim: int = 10825,
    layers: int = 128,
    sample_limit: int = 50000,
    noise_level: float = 0.01,
    gradient_clip: float = 1.0,
) -> None:
    param_count = layers * 2 * dim * dim
    sep = "=" * 70
    print(sep)
    print("JAIDE v40 — RSF Training on Modal B200 GPUs")
    print(sep)
    print(f"GPU:            {DEFAULT_GPU_CONFIG}")
    print(f"Epochs:         {epochs}")
    print(f"Batch Size:     {batch_size}")
    print(f"Learning Rate:  {learning_rate}")
    print(f"Embedding Dim:  {dim}")
    print(f"RSF Layers:     {layers}")
    print(f"Parameters:     ~{param_count / 1e9:.1f}B")
    print(f"Sample Limit:   {sample_limit}")
    print(f"Noise Level:    {noise_level}")
    print(f"Gradient Clip:  {gradient_clip}")
    print(sep)

    result = train_jaide_rsf.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dim=dim,
        layers=layers,
        sample_limit=sample_limit,
        noise_level=noise_level,
        gradient_clip=gradient_clip,
    )

    print(f"\n{sep}")
    print("TRAINING RESULT")
    print(sep)
    print(f"Status:   {result.get('status')}")
    dur = result.get("duration_seconds")
    if dur is not None:
        print(f"Duration: {float(dur):.1f}s  ({float(dur)/60:.1f} min)")
    print(f"GPU:      {result.get('gpu_config')}")
    mp = result.get("model_path")
    if mp:
        print(f"Model:    {mp}")
    out = result.get("stdout", "")
    err = result.get("stderr", "")
    if out:
        print("\n--- stdout ---")
        print(out[-6000:])
    if err:
        print("\n--- stderr ---")
        print(err[-3000:])
    print(sep)

    if result.get("status") == "completed":
        print("\nListing saved models...")
        info = list_models.remote()
        for m in info.get("models", [])[:5]:
            print(f"  {m['filename']} ({m['size_mb']:.1f} MB)")
