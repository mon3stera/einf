from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def cute_gemm_op(A, B):
    return torch.ops.einf.cute_gemm(A, B)


DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/einf/executors/torch/ops/csrc/cute_gemm_cuda.cu"
)


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def tflops(m: int, n: int, k: int, latency_us: float) -> float:
    return 2.0 * m * n * k / (latency_us * 1e6)


def cuda_time_us(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def load_gemm_only(verbose: bool) -> None:
    project_root = Path(__file__).resolve().parents[1]
    csrc = project_root / "src/einf/executors/torch/ops/csrc"
    cutlass_include = project_root / "third_party/cutlass/include"
    load(
        name="einf_gemm_only",
        sources=[str(csrc / "cute_gemm.cpp"), str(csrc / "cute_gemm_cuda.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas", "-v"],
        extra_include_paths=[str(cutlass_include)],
        with_cuda=True,
        is_python_module=False,
        verbose=verbose,
    )


def dump_kernel_resources() -> None:
    cache = Path(os.environ.get("TORCH_EXTENSIONS_DIR", Path.home() / ".cache/torch_extensions"))
    cubins = list(cache.rglob("einf_gemm_only*.so")) + list(cache.rglob("cute_gemm_cuda.cuda.o"))
    nvcc = os.environ.get("CUDA_HOME", "/usr/local/cuda") + "/bin/cuobjdump"
    targets = list(cache.rglob("*cute_gemm*"))
    print("extension artifacts:")
    for path in targets[:20]:
        print(f"  {path}")
    obj = next((p for p in targets if p.suffix == ".o" and "cute_gemm_cuda" in p.name), None)
    if obj is None or not Path(nvcc).is_file():
        return
    completed = subprocess.run(
        [nvcc, "--dump-resource-usage", str(obj)],
        check=False,
        capture_output=True,
        text=True,
    )
    print(completed.stdout or completed.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--shapes", default="2048x2048x2048,4096x4096x4096")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch: {torch.__version__} cuda: {torch.version.cuda}")
    print(f"source: {args.source} hash={source_hash(args.source)}")
    load_gemm_only(args.verbose_build)
    dump_kernel_resources()

    print(
        f"{'MxNxK':>16} {'cute_us':>10} {'cute_TF':>8} "
        f"{'ieee_us':>10} {'ieee_TF':>8} {'ieee_%':>7} "
        f"{'tf32_us':>10} {'tf32_TF':>8} {'tf32_%':>7}"
    )

    for raw in args.shapes.split(","):
        m, n, k = (int(dim) for dim in raw.lower().split("x"))
        torch.manual_seed(0)
        A = torch.randn((m, k), device="cuda", dtype=torch.float32)
        B = torch.randn((k, n), device="cuda", dtype=torch.float32)
        C_ref = None
        prev = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            C_ref = A @ B
            C = cute_gemm_op(A, B)
            torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=5e-4)
            cute_us = cuda_time_us(
                lambda: cute_gemm_op(A, B), warmup=args.warmup, iters=args.iters
            )
            ieee_us = cuda_time_us(lambda: A @ B, warmup=args.warmup, iters=args.iters)
        finally:
            torch.set_float32_matmul_precision(prev)

        torch.set_float32_matmul_precision("high")
        try:
            tf32_us = cuda_time_us(lambda: A @ B, warmup=args.warmup, iters=args.iters)
        finally:
            torch.set_float32_matmul_precision(prev)

        print(
            f"{f'{m}x{n}x{k}':>16} "
            f"{cute_us:10.1f} {tflops(m, n, k, cute_us):8.2f} "
            f"{ieee_us:10.1f} {tflops(m, n, k, ieee_us):8.2f} "
            f"{100.0 * ieee_us / cute_us:6.1f}% "
            f"{tf32_us:10.1f} {tflops(m, n, k, tf32_us):8.2f} "
            f"{100.0 * tf32_us / cute_us:6.1f}%"
        )


if __name__ == "__main__":
    main()
