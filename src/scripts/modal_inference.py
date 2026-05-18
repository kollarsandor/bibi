import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

app = modal.App("jaide-v40-inference")

LOCAL_SRC_DIR = (Path(__file__).resolve().parent / "..").resolve()

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        "DEBIAN_FRONTEND=noninteractive apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-change-held-packages "
        "build-essential wget curl ca-certificates xz-utils libgomp1 opam ocaml",
        "rm -rf /var/lib/apt/lists/*",
    )
    .run_commands(
        "curl -sSf https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz -o /tmp/zig.tar.xz",
        "tar -xf /tmp/zig.tar.xz -C /tmp",
        "mv /tmp/zig-linux-x86_64-0.13.0 /usr/local/zig",
        "ln -sf /usr/local/zig/zig /usr/local/bin/zig",
        "zig version",
        "opam init --disable-sandboxing -y",
        "eval $(opam env) && opam install -y futhark || true",
    )
    .add_local_dir(
        str(LOCAL_SRC_DIR),
        remote_path="/jaide_src",
        copy=True,
    )
    .run_commands(
        "eval $(opam env) && futhark c /jaide_src/hw/accel/futhark_kernels.fut "
        "-o /jaide_src/hw/accel/futhark_kernels.c 2>&1 | tail -5 || true",
        "zig build-exe /jaide_src/main.zig --main-pkg-path /jaide_src "
        "-O ReleaseFast -fstrip -femit-bin=/root/jaide -lc 2>&1 | tail -20 || echo 'Will build at runtime'",
        "ls /root/jaide 2>/dev/null && echo 'Inference binary ready' || echo 'Will build at runtime'",
    )
)

volume = modal.Volume.from_name("jaide-training-data", create_if_missing=True)
models_volume = modal.Volume.from_name("jaide-training-data", create_if_missing=True)

MODELS_DIR = Path("/models")
BINARY_PATH = Path("/root/jaide")
SRC_DIR = Path("/jaide_src")


def _runtime_build_inference() -> None:
    if BINARY_PATH.is_file():
        return
    import subprocess
    futhark_c = SRC_DIR / "hw/accel/futhark_kernels.c"
    if not futhark_c.is_file():
        subprocess.run(
            ["bash", "-c", f"eval $(opam env) && futhark c {SRC_DIR}/hw/accel/futhark_kernels.fut -o {futhark_c}"],
            capture_output=True,
        )
    r = subprocess.run(
        ["zig", "build-exe", str(SRC_DIR / "main.zig"),
         "--main-pkg-path", str(SRC_DIR),
         "-O", "ReleaseFast", "-fstrip",
         f"-femit-bin={BINARY_PATH}", "-lc"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Inference build failed:\n{r.stderr[:4000]}")


@app.function(
    image=image,
    gpu="B200:1",
    volumes={str(MODELS_DIR): models_volume},
    cpu=16,
    memory=65536,
    timeout=3600,
)
def inference(
    prompt: str,
    model_filename: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    import subprocess

    models_volume.reload()
    _runtime_build_inference()

    model_path = MODELS_DIR / Path(model_filename).name
    if not model_path.is_file():
        return {
            "status": "model_not_found",
            "error": f"Model not found: {model_filename}",
            "prompt": prompt,
            "output": None,
            "inference_time_seconds": 0.0,
        }

    infer_args = [
        str(BINARY_PATH),
        "--mode", "infer",
        "--model", str(model_path),
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    start_time = time.time()
    result = subprocess.run(infer_args, capture_output=True, text=True, env=env)
    end_time = time.time()

    return {
        "status": "completed" if result.returncode == 0 else "failed",
        "prompt": prompt,
        "output": result.stdout,
        "error": result.stderr if result.returncode != 0 else None,
        "inference_time_seconds": end_time - start_time,
        "max_tokens": max_tokens,
        "model": model_filename,
    }


@app.function(
    image=image,
    gpu="B200:1",
    volumes={str(MODELS_DIR): models_volume},
    cpu=16,
    memory=65536,
    timeout=7200,
)
def batch_inference(
    prompts: List[str],
    model_filename: str,
    max_tokens: int = 256,
) -> Dict[str, Any]:
    import subprocess

    models_volume.reload()
    _runtime_build_inference()

    model_path = MODELS_DIR / Path(model_filename).name
    if not model_path.is_file():
        return {"status": "model_not_found", "results": [], "total_prompts": len(prompts)}

    results = []
    total_start = time.time()

    for idx, prompt in enumerate(prompts):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        infer_args = [
            str(BINARY_PATH),
            "--mode", "infer",
            "--model", str(model_path),
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
        ]
        t0 = time.time()
        r = subprocess.run(infer_args, capture_output=True, text=True, env=env)
        results.append({
            "index": idx,
            "prompt": prompt,
            "status": "completed" if r.returncode == 0 else "failed",
            "output": r.stdout,
            "error": r.stderr if r.returncode != 0 else None,
            "time_seconds": time.time() - t0,
        })

    total_end = time.time()
    return {
        "total_prompts": len(prompts),
        "total_time_seconds": total_end - total_start,
        "average_time_per_prompt": (total_end - total_start) / max(len(prompts), 1),
        "results": results,
    }


@app.local_entrypoint()
def main(
    prompt: str = "Mi az általános relativitáselmélet lényege?",
    model: Optional[str] = None,
    max_tokens: int = 256,
) -> None:
    from modal_train import list_models

    if model is None:
        print("Fetching latest model from volume...")
        models_info = list_models.remote()
        model_list = models_info.get("models", [])
        if not model_list:
            print("No trained models found. Run training first:")
            print("  modal run modal_train.py")
            return
        model = model_list[0]["filename"]
        print(f"Using: {model}")

    sep = "=" * 70
    print(sep)
    print("JAIDE v40 — Inference on B200 GPU")
    print(sep)
    print(f"Model:      {model}")
    print(f"Prompt:     {prompt}")
    print(f"Max Tokens: {max_tokens}")
    print(sep)

    result = inference.remote(
        prompt=prompt,
        model_filename=model,
        max_tokens=max_tokens,
    )

    print(f"\n{sep}")
    print("RESULT")
    print(sep)
    print(f"Status: {result['status']}")
    print(f"Time:   {result['inference_time_seconds']:.3f}s")
    if result["status"] == "completed":
        print(f"\nPrompt: {result['prompt']}")
        print(f"\nOutput:\n{result['output']}")
    else:
        print(f"\nError: {result.get('error')}")
    print(sep)
