from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from einf.executors.torch.ops import cute_gemm, load_custom_ops


DEFAULT_CUTLASS_PROFILER = Path(
    os.environ.get(
        "CUTLASS_PROFILER",
        "/tmp/einf-cutlass-build/tools/profiler/cutlass_profiler",
    )
)
DEFAULT_CUTE_GEMM_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/einf/executors/torch/ops/csrc/cute_gemm_cuda.cu"
)


@dataclass(frozen=True, slots=True)
class GemmCase:
    M: int
    N: int
    K: int

    @property
    def name(self) -> str:
        return f"{self.M}x{self.N}x{self.K}"


@dataclass(frozen=True, slots=True)
class GemmKernelConfig:
    cta_m: int
    cta_n: int
    cta_k: int
    stages: int
    threads: int
    warps_m: int | None = None
    warps_n: int | None = None
    warps_k: int | None = None
    inst_m: int | None = None
    inst_n: int | None = None
    inst_k: int | None = None
    raster_order: str | None = None
    swizzle_size: int | None = None

    @property
    def cta(self) -> str:
        return f"{self.cta_m}x{self.cta_n}x{self.cta_k}"

    @property
    def warps(self) -> str:
        if self.warps_m is None or self.warps_n is None or self.warps_k is None:
            return "unknown"
        return f"{self.warps_m}x{self.warps_n}x{self.warps_k}"

    @property
    def instruction(self) -> str:
        if self.inst_m is None or self.inst_n is None or self.inst_k is None:
            return "unknown"
        return f"{self.inst_m}x{self.inst_n}x{self.inst_k}"

    def summary(self) -> str:
        details = [
            f"cta={self.cta}",
            f"stages={self.stages}",
            f"threads={self.threads}",
        ]
        if self.warps_m is not None:
            details.append(f"warps={self.warps}")
        if self.inst_m is not None:
            details.append(f"inst={self.instruction}")
        if self.raster_order:
            details.append(f"raster={self.raster_order}")
        if self.swizzle_size is not None:
            details.append(f"swizzle={self.swizzle_size}")
        return " ".join(details)


@dataclass(frozen=True, slots=True)
class CutlassResult:
    latency_us: float
    kernel: str
    config: GemmKernelConfig


DEFAULT_CASES = (
    GemmCase(128, 128, 128),
    GemmCase(256, 256, 256),
    GemmCase(512, 512, 512),
    GemmCase(1024, 1024, 1024),
    GemmCase(2048, 2048, 2048),
    GemmCase(4096, 4096, 4096),
    GemmCase(128, 512, 512),
    GemmCase(512, 2048, 512),
    GemmCase(512, 512, 2048),
    GemmCase(508, 508, 516),
)

QUICK_CASES = (
    GemmCase(128, 128, 128),
    GemmCase(512, 512, 512),
    GemmCase(1024, 1024, 1024),
)

CUTLASS_SIMT_KERNELS = "cutlass_simt_sgemm_*_nn_align1"
CUTLASS_TF32_KERNELS = "cutlass_tensorop_s1688gemm_tf32_*_nn_align*"


def parse_source_constant(source: str, name: str) -> int:
    match = re.search(rf"constexpr\s+int\s+{re.escape(name)}\s*=\s*(\d+)\s*;", source)
    if match is None:
        raise ValueError(f"could not find integer {name} in cute_gemm source")
    return int(match.group(1))


def read_cute_config(source_path: Path) -> GemmKernelConfig:
    source = source_path.read_text()
    return GemmKernelConfig(
        cta_m=parse_source_constant(source, "kBlockM"),
        cta_n=parse_source_constant(source, "kBlockN"),
        cta_k=parse_source_constant(source, "kBlockK"),
        stages=parse_source_constant(source, "kStages"),
        threads=parse_source_constant(source, "kNumThreads"),
    )


def cutlass_config_from_row(row: dict[str, str]) -> GemmKernelConfig:
    warps_m = int(row["warps_m"])
    warps_n = int(row["warps_n"])
    warps_k = int(row["warps_k"])
    return GemmKernelConfig(
        cta_m=int(row["cta_m"]),
        cta_n=int(row["cta_n"]),
        cta_k=int(row["cta_k"]),
        stages=int(row["stages"]),
        threads=32 * warps_m * warps_n * warps_k,
        warps_m=warps_m,
        warps_n=warps_n,
        warps_k=warps_k,
        inst_m=int(row["inst_m"]),
        inst_n=int(row["inst_n"]),
        inst_k=int(row["inst_k"]),
        raster_order=row["raster_order"],
        swizzle_size=int(row["swizzle_size"]),
    )


def parse_cases(value: str) -> tuple[GemmCase, ...]:
    cases: list[GemmCase] = []
    for raw_case in value.split(","):
        dimensions = raw_case.lower().split("x")
        if len(dimensions) != 3:
            raise argparse.ArgumentTypeError(
                f"invalid GEMM shape {raw_case!r}; expected MxNxK"
            )
        try:
            M, N, K = (int(dimension) for dimension in dimensions)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid GEMM shape {raw_case!r}; dimensions must be integers"
            ) from error
        if min(M, N, K) <= 0:
            raise argparse.ArgumentTypeError(
                f"invalid GEMM shape {raw_case!r}; dimensions must be positive"
            )
        if M % 4 != 0 or N % 4 != 0 or K % 4 != 0:
            raise argparse.ArgumentTypeError(
                f"invalid GEMM shape {raw_case!r}; M, N, and K must be divisible by 4"
            )
        cases.append(GemmCase(M, N, K))
    return tuple(cases)


def benchmark_us(
    function: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def tflops(case: GemmCase, latency_us: float) -> float:
    return 2.0 * case.M * case.N * case.K / (latency_us * 1e6)


def run_cutlass_profiler(
    profiler: Path,
    case: GemmCase,
    *,
    kernels: str,
    warmup: int,
    iterations: int,
    output_directory: Path,
    label: str,
) -> CutlassResult:
    output_base = output_directory / f"{case.name}-{label}"

    # Row-major C=A@B is memory-equivalent to the column-major transposed
    # problem C^T=B^T@A^T. CUTLASS therefore receives (N,M,K) and its NN
    # column-major kernels, preserving the same contiguous A/B/output storage.
    command = [
        str(profiler),
        "--operation=Gemm",
        f"--kernels={kernels}",
        f"--m={case.N}",
        f"--n={case.M}",
        f"--k={case.K}",
        "--alpha=1",
        "--beta=0",
        "--verification-enabled=false",
        f"--warmup-iterations={warmup}",
        f"--profiling-iterations={iterations}",
        "--workspace-count=1",
        f"--output={output_base}",
        "--verbose=false",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CUTLASS profiler failed:\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    csv_path = Path(f"{output_base}.gemm.csv")
    with csv_path.open(newline="") as csv_file:
        rows = [
            row
            for row in csv.DictReader(csv_file)
            if row["Status"].lower() == "success" and row["Runtime"]
        ]
    if not rows:
        raise RuntimeError(
            f"CUTLASS profiler found no successful {label} kernel for {case.name}"
        )

    best = min(rows, key=lambda row: float(row["Runtime"]))
    return CutlassResult(
        latency_us=float(best["Runtime"]) * 1000,
        kernel=best["Operation"],
        config=cutlass_config_from_row(best),
    )


def make_inputs(case: GemmCase) -> tuple[torch.Tensor, torch.Tensor]:
    A = torch.randn((case.M, case.K), device="cuda", dtype=torch.float32)
    B = torch.randn((case.K, case.N), device="cuda", dtype=torch.float32)
    return A, B


def verify(A: torch.Tensor, B: torch.Tensor) -> None:
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        expected = A @ B
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    torch.testing.assert_close(cute_gemm(A, B), expected, rtol=1e-4, atol=5e-4)


def print_selected_kernels(
    results: Sequence[tuple[GemmCase, CutlassResult, CutlassResult | None]],
    cute_config: GemmKernelConfig,
) -> None:
    print(f"\neinf CuTe configuration: {cute_config.summary()}")
    print("selected CUTLASS kernels and configurations:")
    for case, simt, tf32 in results:
        print(
            f"  {case.name} FP32 SIMT: {simt.config.summary()} | {simt.kernel}"
        )
        if tf32 is not None:
            print(
                f"  {case.name} TF32 Tensor Core: "
                f"{tf32.config.summary()} | {tf32.kernel}"
            )

    grouped_cases: dict[GemmKernelConfig, list[str]] = {}
    for case, simt, _ in results:
        grouped_cases.setdefault(simt.config, []).append(case.name)

    print("\nunique selected FP32 SIMT configurations:")
    for config, case_names in grouped_cases.items():
        print(f"  {config.summary()}: {', '.join(case_names)}")

    exact_matches = [
        case.name
        for case, simt, _ in results
        if (
            simt.config.cta_m,
            simt.config.cta_n,
            simt.config.cta_k,
            simt.config.stages,
        )
        == (
            cute_config.cta_m,
            cute_config.cta_n,
            cute_config.cta_k,
            cute_config.stages,
        )
    ]
    if exact_matches:
        print(
            "\nCUTLASS selected the same CTA tile and stage count as einf for: "
            + ", ".join(exact_matches)
        )
    else:
        print(
            "\nCUTLASS did not select the same CTA tile and stage count as the "
            "current einf kernel for any measured shape."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark einf's FP32 CuTe GEMM against pinned CUTLASS FP32 SIMT "
            "and TF32 Tensor Core kernels"
        )
    )
    parser.add_argument(
        "--cutlass-profiler",
        type=Path,
        default=DEFAULT_CUTLASS_PROFILER,
    )
    parser.add_argument(
        "--cute-gemm-source",
        type=Path,
        default=DEFAULT_CUTE_GEMM_SOURCE,
        help="source file used to report einf's compile-time CTA/stage configuration",
    )
    parser.add_argument(
        "--shapes",
        type=parse_cases,
        help="comma-separated MxNxK cases; overrides the default matrix",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="benchmark only 128^3, 512^3, and 1024^3",
    )
    parser.add_argument(
        "--skip-tf32",
        action="store_true",
        help="omit the non-IEEE-FP32 Tensor Core performance ceiling",
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if not args.cutlass_profiler.is_file():
        raise FileNotFoundError(
            f"CUTLASS profiler not found at {args.cutlass_profiler}; run "
            "scripts/build_cutlass_profiler.sh first or pass --cutlass-profiler"
        )
    if not args.cute_gemm_source.is_file():
        raise FileNotFoundError(f"cute_gemm source not found: {args.cute_gemm_source}")

    cases = args.shapes or (QUICK_CASES if args.quick else DEFAULT_CASES)
    cute_config = read_cute_config(args.cute_gemm_source)
    torch.manual_seed(0)
    load_custom_ops()

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}")
    print(f"warmup={args.warmup}, iterations={args.iterations}, workspace_count=1")
    print(f"einf CuTe: {cute_config.summary()}")
    print("CUTLASS runs the transposed NN column-major equivalent of row-major GEMM.")
    if not args.skip_tf32:
        print("TF32 Tensor Core is a performance ceiling, not exact IEEE FP32 arithmetic.")
    print(
        f"{'MxNxK':>16} {'cute_us':>10} {'cute_TF':>9} "
        f"{'simt_us':>10} {'simt_TF':>9} {'simt_%':>8} "
        f"{'tf32_us':>10} {'tf32_TF':>9} {'tf32_%':>8}"
    )

    selected_kernels: list[
        tuple[GemmCase, CutlassResult, CutlassResult | None]
    ] = []
    with torch.inference_mode(), tempfile.TemporaryDirectory(
        prefix="einf-cute-gemm-"
    ) as temporary_directory:
        output_directory = Path(temporary_directory)
        for case in cases:
            A, B = make_inputs(case)
            verify(A, B)

            cute_latency_us = benchmark_us(
                lambda: cute_gemm(A, B),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            simt = run_cutlass_profiler(
                args.cutlass_profiler,
                case,
                kernels=CUTLASS_SIMT_KERNELS,
                warmup=args.warmup,
                iterations=args.iterations,
                output_directory=output_directory,
                label="simt",
            )
            tf32 = None
            if not args.skip_tf32:
                tf32 = run_cutlass_profiler(
                    args.cutlass_profiler,
                    case,
                    kernels=CUTLASS_TF32_KERNELS,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    output_directory=output_directory,
                    label="tf32",
                )

            simt_share = 100.0 * simt.latency_us / cute_latency_us
            if tf32 is None:
                tf32_latency = tf32_throughput = tf32_share = float("nan")
            else:
                tf32_latency = tf32.latency_us
                tf32_throughput = tflops(case, tf32.latency_us)
                tf32_share = 100.0 * tf32.latency_us / cute_latency_us

            print(
                f"{case.name:>16} "
                f"{cute_latency_us:10.3f} {tflops(case, cute_latency_us):9.2f} "
                f"{simt.latency_us:10.3f} {tflops(case, simt.latency_us):9.2f} "
                f"{simt_share:7.1f}% "
                f"{tf32_latency:10.3f} {tf32_throughput:9.2f} "
                f"{tf32_share:7.1f}%"
            )
            selected_kernels.append((case, simt, tf32))

    print_selected_kernels(selected_kernels, cute_config)


if __name__ == "__main__":
    main()
