from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right


DEFAULT_HEAD_DIM = 64
DEFAULT_CUTE_BLOCK_M = 64
DEFAULT_CUTE_BLOCK_N = 16
DEFAULT_CUTE_NUM_WARPS = 4
DEFAULT_STAGES = 2
SM80_MMA_M = 16
SM80_MMA_N = 8
SM80_MMA_K = 16
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUTE_SOURCE = PROJECT_ROOT / "src/einf/executors/torch/dsl/flash_attention.py"


@dataclass(frozen=True, slots=True)
class AttentionCase:
    q_len: int
    kv_len: int

    @property
    def name(self) -> str:
        if self.q_len == self.kv_len:
            return f"prefill-{self.q_len}"
        return f"chunk-{self.q_len}/{self.kv_len}"

    @property
    def start_pos(self) -> int:
        return self.kv_len - self.q_len


@dataclass(frozen=True, slots=True)
class CaseResult:
    case: AttentionCase
    cute_us: tuple[float, ...]
    sdpa_us: tuple[float, ...]
    orders: tuple[str, ...]
    max_abs_diff: float
    compile_seconds: float

    @property
    def cute_median_us(self) -> float:
        return statistics.median(self.cute_us)

    @property
    def sdpa_median_us(self) -> float:
        return statistics.median(self.sdpa_us)

    @property
    def paired_ratios(self) -> tuple[float, ...]:
        # Greater than one means CuTe is faster than production Flash SDPA.
        return tuple(sdpa / cute for cute, sdpa in zip(self.cute_us, self.sdpa_us))

    @property
    def paired_ratio_median(self) -> float:
        return statistics.median(self.paired_ratios)


@dataclass(frozen=True, slots=True)
class GridCell:
    case: AttentionCase
    block_m: int
    block_n: int
    num_warps: int
    stages: int
    cute_us: tuple[float, ...]
    max_abs_diff: float
    compile_seconds: float
    error: str | None = None

    @property
    def config_name(self) -> str:
        return (
            f"{self.block_m}x{self.block_n}-w{self.num_warps}-s{self.stages}"
        )

    @property
    def cute_median_us(self) -> float:
        return statistics.median(self.cute_us) if self.cute_us else math.inf


DEFAULT_CASES = (
    AttentionCase(64, 64),
    AttentionCase(128, 128),
    AttentionCase(512, 512),
    AttentionCase(2048, 2048),
    AttentionCase(4096, 4096),
    AttentionCase(8192, 8192),
    AttentionCase(16384, 16384),
    AttentionCase(64, 512),
    AttentionCase(64, 2048),
    AttentionCase(64, 8192),
    AttentionCase(128, 2048),
)

PREFILL_GRID_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
DEFAULT_GRID_BLOCK_M = (64, 128, 256)
DEFAULT_GRID_BLOCK_N = (16, 32, 64)
DEFAULT_GRID_NUM_WARPS = (2, 4)
DEFAULT_GRID_STAGES = (1, 2, 3)

QUICK_CASES = (
    AttentionCase(64, 64),
    AttentionCase(128, 128),
    AttentionCase(64, 512),
)


def validate_tile_config(
    block_m: int,
    block_n: int,
    head_dim: int,
    num_warps: int,
) -> None:
    if block_m <= 0 or block_m % SM80_MMA_M != 0:
        raise ValueError("CuTe block_m must be a positive multiple of 16")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("CuTe num_warps must be one of 1, 2, 4, or 8")
    if block_m % num_warps != 0 or (block_m // num_warps) % SM80_MMA_M != 0:
        raise ValueError("CuTe block_m / num_warps must be a multiple of 16")
    if block_n <= 0 or block_n % SM80_MMA_K != 0:
        raise ValueError("CuTe block_n must be a positive multiple of 16")
    if head_dim < 64:
        raise ValueError("CuTe head_dim must be at least 64")
    if head_dim % SM80_MMA_K != 0:
        raise ValueError("CuTe head_dim must be a multiple of 16")
    if head_dim % 8 != 0:
        raise ValueError("CuTe head_dim must be divisible by 8 for 128-bit BF16 copies")
    copy_elements_per_cta = 32 * num_warps * 8
    if (block_m * head_dim) % copy_elements_per_cta != 0:
        raise ValueError(
            "CuTe Q tile must divide evenly across CTA threads in "
            "128-bit BF16 copies"
        )
    if (block_n * head_dim) % copy_elements_per_cta != 0:
        raise ValueError(
            "CuTe K/V tile must divide evenly across CTA threads in "
            "128-bit BF16 copies"
        )
    if (block_n // SM80_MMA_N) % 2 != 0:
        raise ValueError("CuTe QK MMA_N repeats must pair into P@V K=16 atoms")


def validate_case_lengths(case: AttentionCase) -> None:
    if case.q_len <= 0 or case.kv_len <= 0:
        raise ValueError("q_len and kv_len must be positive")
    if case.kv_len < case.q_len:
        raise ValueError(
            f"kv_len must be at least q_len, got {case.q_len}x{case.kv_len}"
        )


def validate_case(
    case: AttentionCase,
    *,
    block_m: int = DEFAULT_CUTE_BLOCK_M,
    block_n: int = DEFAULT_CUTE_BLOCK_N,
) -> None:
    validate_case_lengths(case)
    if case.q_len % block_m != 0:
        raise ValueError(
            f"CuTe v1 requires q_len divisible by {block_m}, "
            f"got {case.q_len}x{case.kv_len}"
        )
    if case.kv_len % block_n != 0:
        raise ValueError(
            f"CuTe v1 requires kv_len divisible by {block_n}, "
            f"got {case.q_len}x{case.kv_len}"
        )


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid integer list {value!r}"
        ) from error
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def iter_grid_configs(
    *,
    head_dim: int,
    block_ms: Sequence[int],
    block_ns: Sequence[int],
    num_warps_list: Sequence[int],
    stages_list: Sequence[int],
    q_len: int,
    kv_len: int,
) -> tuple[tuple[int, int, int, int], ...]:
    configs: list[tuple[int, int, int, int]] = []
    for block_m in block_ms:
        for block_n in block_ns:
            for num_warps in num_warps_list:
                for stages in stages_list:
                    if stages < 1:
                        continue
                    try:
                        validate_tile_config(block_m, block_n, head_dim, num_warps)
                        validate_case(
                            AttentionCase(q_len, kv_len),
                            block_m=block_m,
                            block_n=block_n,
                        )
                    except ValueError:
                        continue
                    if (block_m // num_warps) > 32:
                        continue
                    configs.append((block_m, block_n, num_warps, stages))
    return tuple(configs)


def grid_iterations(q_len: int) -> int:
    if q_len >= 16384:
        return 8
    if q_len >= 8192:
        return 12
    if q_len >= 4096:
        return 20
    return 30


def parse_cases(value: str) -> tuple[AttentionCase, ...]:
    cases: list[AttentionCase] = []
    for raw_case in value.split(","):
        raw_case = raw_case.strip()
        dimensions = raw_case.lower().split("x")
        if len(dimensions) != 2:
            raise argparse.ArgumentTypeError(
                f"invalid attention shape {raw_case!r}; expected QxKV"
            )
        try:
            q_len, kv_len = (int(dimension) for dimension in dimensions)
            case = AttentionCase(q_len, kv_len)
            validate_case_lengths(case)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
        cases.append(case)
    if not cases:
        raise argparse.ArgumentTypeError("at least one attention shape is required")
    return tuple(cases)


def benchmark_us(
    function: Callable[[], torch.Tensor],
    *,
    iterations: int,
    stream: torch.cuda.Stream,
) -> float:
    """Measure one repeated launch batch with CUDA events on ``stream``."""
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    result: torch.Tensor | None = None

    start.record(stream)
    for _ in range(iterations):
        result = function()
    end.record(stream)
    end.synchronize()

    # Keep the final output alive until the end event has completed.
    del result
    return start.elapsed_time(end) * 1000.0 / iterations


def effective_flops(
    case: AttentionCase,
    *,
    num_attention_heads: int,
    head_dim: int,
) -> int:
    """QK plus P@V FLOPs over mathematically visible lower-right-causal pairs."""
    visible_pairs_per_head = (
        case.q_len * (case.start_pos + 1)
        + case.q_len * (case.q_len - 1) // 2
    )
    return 4 * head_dim * num_attention_heads * visible_pairs_per_head


def effective_tflops(
    case: AttentionCase,
    latency_us: float,
    *,
    num_attention_heads: int,
    head_dim: int,
) -> float:
    return effective_flops(
        case,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
    ) / (latency_us * 1.0e6)


def make_inputs(
    case: AttentionCase,
    *,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    Q = torch.randn(
        (case.q_len, num_attention_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    K = torch.randn(
        (case.kv_len, num_kv_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    V = torch.randn_like(K)
    return Q, K, V


def benchmark_case(
    case: AttentionCase,
    *,
    case_index: int,
    cute_compile: Callable[..., object],
    cute_launch: Callable[..., object],
    from_dlpack: Callable[[torch.Tensor], object],
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_m: int,
    block_n: int,
    num_warps: int,
    stages: int,
    seed: int,
    warmup: int,
    iterations: int,
    rounds: int,
    rtol: float,
    atol: float,
    stream: torch.cuda.Stream,
) -> CaseResult:
    Q, K, V = make_inputs(
        case,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seed=seed + case_index,
    )
    scale = 1.0 / math.sqrt(head_dim)
    causal_bias = causal_lower_right(case.q_len, case.kv_len)
    sdpa_Q = Q.transpose(0, 1).unsqueeze(0)
    sdpa_K = K.transpose(0, 1).unsqueeze(0)
    sdpa_V = V.transpose(0, 1).unsqueeze(0)
    cute_output = torch.empty_like(Q)
    for name, tensor in (
        ("Q", Q),
        ("K", K),
        ("V", V),
        ("O", cute_output),
    ):
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} must be 16-byte aligned for 128-bit copies")
    mQ, mK, mV, mO = (
        from_dlpack(tensor, assumed_align=16)
        for tensor in (Q, K, V, cute_output)
    )

    compile_start = time.perf_counter()
    compiled_cute = cute_compile(
        cute_launch,
        mQ,
        mK,
        mV,
        mO,
        case.start_pos,
        scale,
        block_m,
        block_n,
        head_dim,
        num_warps,
        stages,
    )
    compile_seconds = time.perf_counter() - compile_start

    def run_cute() -> torch.Tensor:
        compiled_cute(mQ, mK, mV, mO, case.start_pos, scale)
        return cute_output

    def run_sdpa() -> torch.Tensor:
        output = F.scaled_dot_product_attention(
            sdpa_Q,
            sdpa_K,
            sdpa_V,
            attn_mask=causal_bias,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )
        return output.squeeze(0).transpose(0, 1)

    # CuTe compilation happens above. Correctness and all warmups are outside timing.
    with torch.cuda.stream(stream), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        run_cute()
        sdpa_output = run_sdpa()
    stream.synchronize()
    if (
        cute_output.shape != sdpa_output.shape
        or cute_output.dtype != sdpa_output.dtype
    ):
        raise AssertionError(
            "CuTe/Flash-SDPA output metadata differs: "
            f"CuTe={cute_output.shape}/{cute_output.dtype}, "
            f"SDPA={sdpa_output.shape}/{sdpa_output.dtype}"
        )
    max_abs_diff = (cute_output.float() - sdpa_output.float()).abs().max().item()
    torch.testing.assert_close(cute_output, sdpa_output, rtol=rtol, atol=atol)
    del sdpa_output

    # Give each implementation the same number of interleaved warmup launches.
    with torch.cuda.stream(stream), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for warmup_index in range(warmup):
            if (warmup_index + case_index) % 2 == 0:
                run_cute()
                run_sdpa()
            else:
                run_sdpa()
                run_cute()
    stream.synchronize()

    cute_samples: list[float] = []
    sdpa_samples: list[float] = []
    orders: list[str] = []
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for round_index in range(rounds):
            cute_first = (round_index + case_index) % 2 == 0
            order = (
                (("cute", run_cute), ("sdpa", run_sdpa))
                if cute_first
                else (("sdpa", run_sdpa), ("cute", run_cute))
            )
            orders.append("cute-sdpa" if cute_first else "sdpa-cute")
            round_samples: dict[str, float] = {}
            with torch.cuda.stream(stream):
                for name, function in order:
                    round_samples[name] = benchmark_us(
                        function,
                        iterations=iterations,
                        stream=stream,
                    )
            cute_samples.append(round_samples["cute"])
            sdpa_samples.append(round_samples["sdpa"])

    return CaseResult(
        case=case,
        cute_us=tuple(cute_samples),
        sdpa_us=tuple(sdpa_samples),
        orders=tuple(orders),
        max_abs_diff=max_abs_diff,
        compile_seconds=compile_seconds,
    )


def benchmark_grid_cell(
    case: AttentionCase,
    *,
    case_index: int,
    cute_compile: Callable[..., object],
    cute_launch: Callable[..., object],
    from_dlpack: Callable[[torch.Tensor], object],
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_m: int,
    block_n: int,
    num_warps: int,
    stages: int,
    seed: int,
    warmup: int,
    iterations: int,
    rounds: int,
    rtol: float,
    atol: float,
    stream: torch.cuda.Stream,
) -> GridCell:
    Q, K, V = make_inputs(
        case,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seed=seed + case_index,
    )
    scale = 1.0 / math.sqrt(head_dim)
    causal_bias = causal_lower_right(case.q_len, case.kv_len)
    sdpa_Q = Q.transpose(0, 1).unsqueeze(0)
    sdpa_K = K.transpose(0, 1).unsqueeze(0)
    sdpa_V = V.transpose(0, 1).unsqueeze(0)
    cute_output = torch.empty_like(Q)
    mQ, mK, mV, mO = (
        from_dlpack(tensor, assumed_align=16)
        for tensor in (Q, K, V, cute_output)
    )

    compile_start = time.perf_counter()
    compiled_cute = cute_compile(
        cute_launch,
        mQ,
        mK,
        mV,
        mO,
        case.start_pos,
        scale,
        block_m,
        block_n,
        head_dim,
        num_warps,
        stages,
    )
    compile_seconds = time.perf_counter() - compile_start

    def run_cute() -> torch.Tensor:
        compiled_cute(mQ, mK, mV, mO, case.start_pos, scale)
        return cute_output

    def run_sdpa() -> torch.Tensor:
        output = F.scaled_dot_product_attention(
            sdpa_Q,
            sdpa_K,
            sdpa_V,
            attn_mask=causal_bias,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )
        return output.squeeze(0).transpose(0, 1)

    with torch.cuda.stream(stream), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        run_cute()
        sdpa_output = run_sdpa()
    stream.synchronize()
    max_abs_diff = (cute_output.float() - sdpa_output.float()).abs().max().item()
    if max_abs_diff > max(atol, rtol):
        raise ValueError(
            f"CuTe vs SDPA max_abs_diff={max_abs_diff:.6g} exceeds "
            f"atol/rtol={max(atol, rtol)}"
        )
    del sdpa_output

    with torch.cuda.stream(stream):
        for _ in range(warmup):
            run_cute()
    stream.synchronize()

    cute_samples: list[float] = []
    with torch.cuda.stream(stream):
        for _ in range(rounds):
            cute_samples.append(
                benchmark_us(run_cute, iterations=iterations, stream=stream)
            )
    return GridCell(
        case=case,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        stages=stages,
        cute_us=tuple(cute_samples),
        max_abs_diff=max_abs_diff,
        compile_seconds=compile_seconds,
    )


def print_grid_summary(cells: Sequence[GridCell]) -> None:
    print("\nBest full-prefill config per length (CuTe median microseconds)")
    print(
        f"{'length':>8} {'best':>18} {'cute_med':>10} {'range':>19} "
        f"{'max_abs':>10} {'n_ok':>5} {'n_fail':>6}"
    )
    lengths = sorted({cell.case.q_len for cell in cells})
    for length in lengths:
        group = [cell for cell in cells if cell.case.q_len == length]
        ok = [cell for cell in group if cell.error is None]
        failed = [cell for cell in group if cell.error is not None]
        if not ok:
            print(
                f"{length:8d} {'FAIL':>18} {'inf':>10} {'':>19} "
                f"{'':>10} {0:5d} {len(failed):6d}"
            )
            continue
        best = min(ok, key=lambda cell: cell.cute_median_us)
        print(
            f"{length:8d} {best.config_name:>18} "
            f"{best.cute_median_us:10.3f} {format_range(best.cute_us):>19} "
            f"{best.max_abs_diff:10.4g} {len(ok):5d} {len(failed):6d}"
        )

    print("\nTop 3 configs per length")
    for length in lengths:
        ok = [
            cell
            for cell in cells
            if cell.case.q_len == length and cell.error is None
        ]
        ok.sort(key=lambda cell: cell.cute_median_us)
        print(f"  prefill-{length}:")
        for cell in ok[:3]:
            print(
                f"    {cell.config_name:>18}  "
                f"{cell.cute_median_us:10.3f} us  "
                f"range={format_range(cell.cute_us)}"
            )
        if not ok:
            print("    (no successful configs)")


def write_grid_csv(path: Path, cells: Sequence[GridCell]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "case",
                "q_len",
                "kv_len",
                "block_m",
                "block_n",
                "num_warps",
                "stages",
                "m_per_warp",
                "round",
                "cute_us",
                "cute_median_us",
                "max_abs_diff",
                "compile_seconds",
                "error",
            ),
        )
        writer.writeheader()
        for cell in cells:
            if cell.error is not None or not cell.cute_us:
                writer.writerow(
                    {
                        "case": cell.case.name,
                        "q_len": cell.case.q_len,
                        "kv_len": cell.case.kv_len,
                        "block_m": cell.block_m,
                        "block_n": cell.block_n,
                        "num_warps": cell.num_warps,
                        "stages": cell.stages,
                        "m_per_warp": cell.block_m // cell.num_warps,
                        "round": "",
                        "cute_us": "",
                        "cute_median_us": "",
                        "max_abs_diff": "",
                        "compile_seconds": f"{cell.compile_seconds:.9f}",
                        "error": cell.error,
                    }
                )
                continue
            median_us = cell.cute_median_us
            for round_index, cute_us in enumerate(cell.cute_us):
                writer.writerow(
                    {
                        "case": cell.case.name,
                        "q_len": cell.case.q_len,
                        "kv_len": cell.case.kv_len,
                        "block_m": cell.block_m,
                        "block_n": cell.block_n,
                        "num_warps": cell.num_warps,
                        "stages": cell.stages,
                        "m_per_warp": cell.block_m // cell.num_warps,
                        "round": round_index,
                        "cute_us": f"{cute_us:.9f}",
                        "cute_median_us": f"{median_us:.9f}",
                        "max_abs_diff": f"{cell.max_abs_diff:.9f}",
                        "compile_seconds": f"{cell.compile_seconds:.9f}",
                        "error": "",
                    }
                )


def source_sha256(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def format_range(samples: Sequence[float]) -> str:
    return f"[{min(samples):.3f},{max(samples):.3f}]"


def format_samples(samples: Sequence[float]) -> str:
    return "[" + ", ".join(f"{sample:.3f}" for sample in samples) + "]"


def print_summary(
    results: Sequence[CaseResult],
    *,
    num_attention_heads: int,
    head_dim: int,
) -> None:
    print("\nSummary (paired_x = sdpa_us / cute_us; greater than 1 favors CuTe)")
    print(
        f"{'case':>18} {'cute_med':>10} {'cute_range':>19} "
        f"{'sdpa_med':>10} {'sdpa_range':>19} {'paired_x':>9} "
        f"{'cute_TF':>9} {'sdpa_TF':>9}"
    )
    for result in results:
        cute_tflops = effective_tflops(
            result.case,
            result.cute_median_us,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
        )
        sdpa_tflops = effective_tflops(
            result.case,
            result.sdpa_median_us,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
        )
        print(
            f"{result.case.name:>18} "
            f"{result.cute_median_us:10.3f} {format_range(result.cute_us):>19} "
            f"{result.sdpa_median_us:10.3f} "
            f"{format_range(result.sdpa_us):>19} "
            f"{result.paired_ratio_median:9.3f} "
            f"{cute_tflops:9.2f} {sdpa_tflops:9.2f}"
        )

    print("\nRaw paired samples (microseconds per call)")
    for result in results:
        print(f"  {result.case.name}:")
        print(f"    order  = {list(result.orders)}")
        print(f"    cute   = {format_samples(result.cute_us)}")
        print(f"    sdpa   = {format_samples(result.sdpa_us)}")
        print(f"    pair_x = {format_samples(result.paired_ratios)}")
        print(f"    max_abs_diff = {result.max_abs_diff:.7g}")
        print(f"    compile_seconds = {result.compile_seconds:.3f}")


def write_csv(
    path: Path,
    results: Sequence[CaseResult],
    *,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_m: int,
    block_n: int,
    num_warps: int,
    warmup: int,
    iterations: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "case",
                "q_len",
                "kv_len",
                "start_pos",
                "num_attention_heads",
                "num_kv_heads",
                "head_dim",
                "block_m",
                "block_n",
                "num_warps",
                "m_per_warp",
                "warmup",
                "iterations",
                "round",
                "order",
                "cute_us",
                "sdpa_us",
                "sdpa_over_cute",
                "cute_effective_tflops",
                "sdpa_effective_tflops",
                "max_abs_diff",
                "cute_compile_seconds",
            ),
        )
        writer.writeheader()
        for result in results:
            for round_index, (cute_us, sdpa_us, order) in enumerate(
                zip(result.cute_us, result.sdpa_us, result.orders)
            ):
                cute_tflops = effective_tflops(
                    result.case,
                    cute_us,
                    num_attention_heads=num_attention_heads,
                    head_dim=head_dim,
                )
                sdpa_tflops = effective_tflops(
                    result.case,
                    sdpa_us,
                    num_attention_heads=num_attention_heads,
                    head_dim=head_dim,
                )
                writer.writerow(
                    {
                        "case": result.case.name,
                        "q_len": result.case.q_len,
                        "kv_len": result.case.kv_len,
                        "start_pos": result.case.start_pos,
                        "num_attention_heads": num_attention_heads,
                        "num_kv_heads": num_kv_heads,
                        "head_dim": head_dim,
                        "block_m": block_m,
                        "block_n": block_n,
                        "num_warps": num_warps,
                        "m_per_warp": block_m // num_warps,
                        "warmup": warmup,
                        "iterations": iterations,
                        "round": round_index,
                        "order": order,
                        "cute_us": f"{cute_us:.9f}",
                        "sdpa_us": f"{sdpa_us:.9f}",
                        "sdpa_over_cute": f"{sdpa_us / cute_us:.9f}",
                        "cute_effective_tflops": f"{cute_tflops:.9f}",
                        "sdpa_effective_tflops": f"{sdpa_tflops:.9f}",
                        "max_abs_diff": f"{result.max_abs_diff:.9f}",
                        "cute_compile_seconds": f"{result.compile_seconds:.9f}",
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Repeated paired CUDA-event benchmark of the Split-Q CuTe DSL "
            "FlashAttention kernel against forced PyTorch Flash SDPA"
        )
    )
    case_group = parser.add_mutually_exclusive_group()
    case_group.add_argument(
        "--cases",
        type=parse_cases,
        help="comma-separated aligned QxKV cases, for example 64x512,128x2048",
    )
    case_group.add_argument(
        "--quick",
        action="store_true",
        help="run prefill-64, prefill-128, and chunk-64/512",
    )
    parser.add_argument("--num-attention-heads", type=int, default=14)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--block-m", type=int, default=DEFAULT_CUTE_BLOCK_M)
    parser.add_argument("--block-n", type=int, default=DEFAULT_CUTE_BLOCK_N)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--num-warps", type=int, default=DEFAULT_CUTE_NUM_WARPS)
    parser.add_argument("--stages", type=int, default=DEFAULT_STAGES)
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help=(
            "same-session full-prefill tile grid over "
            "block_m/block_n/num_warps/stages; ranks CuTe latency per length"
        ),
    )
    parser.add_argument(
        "--grid-block-m",
        type=parse_int_list,
        default=DEFAULT_GRID_BLOCK_M,
    )
    parser.add_argument(
        "--grid-block-n",
        type=parse_int_list,
        default=DEFAULT_GRID_BLOCK_N,
    )
    parser.add_argument(
        "--grid-num-warps",
        type=parse_int_list,
        default=DEFAULT_GRID_NUM_WARPS,
    )
    parser.add_argument(
        "--grid-stages",
        type=parse_int_list,
        default=DEFAULT_GRID_STAGES,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="interleaved warmup calls per implementation and case",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="calls inside each CUDA-event timing sample",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=9,
        help="paired A/B timing rounds per case (minimum 5)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument(
        "--csv",
        type=Path,
        help="optional destination for one raw paired sample per CSV row",
    )
    args = parser.parse_args()

    if args.grid_search:
        if args.cases is None and not args.quick:
            cases = tuple(
                AttentionCase(length, length) for length in PREFILL_GRID_LENGTHS
            )
        else:
            cases = args.cases or (QUICK_CASES if args.quick else DEFAULT_CASES)
    else:
        cases = args.cases or (QUICK_CASES if args.quick else DEFAULT_CASES)
        validate_tile_config(
            args.block_m,
            args.block_n,
            args.head_dim,
            args.num_warps,
        )
        for case in cases:
            validate_case(case, block_m=args.block_m, block_n=args.block_n)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.num_attention_heads <= 0 or args.num_kv_heads <= 0:
        raise ValueError("head counts must be positive")
    if args.num_attention_heads % args.num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_kv_heads")
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.rounds < 5:
        raise ValueError("rounds must be at least 5 for repeated paired A/B timing")
    if args.rtol < 0.0 or args.atol < 0.0:
        raise ValueError("rtol and atol must be nonnegative")

    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from einf.executors.torch.dsl.flash_attention import _flash_attention_launch

    default_stream = torch.cuda.default_stream()
    print(f"device: {torch.cuda.get_device_name()}")
    print(
        f"torch: {torch.__version__}, torch CUDA: {torch.version.cuda}, "
        f"CuTe DSL: {package_version('nvidia-cutlass-dsl')}"
    )
    if args.grid_search:
        print(
            f"Hq={args.num_attention_heads}, Hkv={args.num_kv_heads}, "
            f"D={args.head_dim}, BF16, grid-search full prefill"
        )
        print(
            f"grid block_m={list(args.grid_block_m)}, "
            f"block_n={list(args.grid_block_n)}, "
            f"num_warps={list(args.grid_num_warps)}, "
            f"stages={list(args.grid_stages)}"
        )
        print(
            "timing: CUDA events, CuTe-only after one SDPA correctness check; "
            "iterations scale down for long sequences"
        )
    else:
        print(
            f"Hq={args.num_attention_heads}, Hkv={args.num_kv_heads}, "
            f"D={args.head_dim}, "
            f"tile={args.block_m}x{args.block_n}, warps={args.num_warps}, "
            f"warp_m={args.block_m // args.num_warps}, BF16, "
            f"warmup={args.warmup}, "
            f"iterations={args.iterations}, "
            f"paired_rounds={args.rounds}"
        )
    print(
        "timing: CUDA events on the default stream; CuTe uses a cute.compile "
        "callable and SDPA is forced to SDPBackend.FLASH_ATTENTION; CuTe "
        "compile/JIT, correctness, and warmup are excluded"
    )
    print(
        "effective TFLOP/s counts QK and P@V over visible causal pairs; "
        "softmax and tile padding are excluded"
    )
    print(f"CuTe source SHA256: {source_sha256(CUTE_SOURCE)}")
    print("SDPA backend: forced SDPBackend.FLASH_ATTENTION")

    if args.grid_search:
        cells: list[GridCell] = []
        with torch.inference_mode():
            for case_index, case in enumerate(cases):
                configs = iter_grid_configs(
                    head_dim=args.head_dim,
                    block_ms=args.grid_block_m,
                    block_ns=args.grid_block_n,
                    num_warps_list=args.grid_num_warps,
                    stages_list=args.grid_stages,
                    q_len=case.q_len,
                    kv_len=case.kv_len,
                )
                iterations = grid_iterations(case.q_len)
                warmup = 8 if args.warmup == 20 else args.warmup
                rounds = 5 if args.rounds == 9 else args.rounds
                print(
                    f"\n[{case_index + 1}/{len(cases)}] {case.name}: "
                    f"{len(configs)} configs, warmup={warmup}, "
                    f"iterations={iterations}, rounds={rounds}"
                )
                for config_index, (block_m, block_n, num_warps, stages) in enumerate(
                    configs
                ):
                    label = f"{block_m}x{block_n}-w{num_warps}-s{stages}"
                    try:
                        cell = benchmark_grid_cell(
                            case,
                            case_index=case_index * 1000 + config_index,
                            cute_compile=cute.compile,
                            cute_launch=_flash_attention_launch,
                            from_dlpack=from_dlpack,
                            num_attention_heads=args.num_attention_heads,
                            num_kv_heads=args.num_kv_heads,
                            head_dim=args.head_dim,
                            block_m=block_m,
                            block_n=block_n,
                            num_warps=num_warps,
                            stages=stages,
                            seed=args.seed,
                            warmup=warmup,
                            iterations=iterations,
                            rounds=rounds,
                            rtol=args.rtol,
                            atol=args.atol,
                            stream=default_stream,
                        )
                    except Exception as error:
                        torch.cuda.empty_cache()
                        cell = GridCell(
                            case=case,
                            block_m=block_m,
                            block_n=block_n,
                            num_warps=num_warps,
                            stages=stages,
                            cute_us=(),
                            max_abs_diff=math.nan,
                            compile_seconds=0.0,
                            error=f"{type(error).__name__}: {error}",
                        )
                    cells.append(cell)
                    if cell.error is None:
                        print(
                            f"  {label:>18}  {cell.cute_median_us:10.3f} us  "
                            f"compile={cell.compile_seconds:.2f}s  "
                            f"max_abs={cell.max_abs_diff:.4g}"
                        )
                    else:
                        print(f"  {label:>18}  FAIL  {cell.error}")
        print_grid_summary(cells)
        csv_path = args.csv or (
            PROJECT_ROOT / "benchmark-results/cute-fa-prefill-grid.csv"
        )
        write_grid_csv(csv_path, cells)
        print(f"\nraw CSV: {csv_path}")
        return

    results: list[CaseResult] = []
    with torch.inference_mode():
        for case_index, case in enumerate(cases):
            print(
                f"\n[{case_index + 1}/{len(cases)}] {case.name}: "
                f"Q={case.q_len}, KV={case.kv_len}, start_pos={case.start_pos}"
            )
            result = benchmark_case(
                case,
                case_index=case_index,
                cute_compile=cute.compile,
                cute_launch=_flash_attention_launch,
                from_dlpack=from_dlpack,
                num_attention_heads=args.num_attention_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                block_m=args.block_m,
                block_n=args.block_n,
                num_warps=args.num_warps,
                stages=args.stages,
                seed=args.seed,
                warmup=args.warmup,
                iterations=args.iterations,
                rounds=args.rounds,
                rtol=args.rtol,
                atol=args.atol,
                stream=default_stream,
            )
            results.append(result)
            print(
                f"  correctness max_abs={result.max_abs_diff:.7g}; "
                f"compile={result.compile_seconds:.3f} s; "
                f"CuTe median={result.cute_median_us:.3f} us; "
                f"Flash SDPA median={result.sdpa_median_us:.3f} us; "
                f"paired SDPA/CuTe={result.paired_ratio_median:.3f}x"
            )

    print_summary(
        results,
        num_attention_heads=args.num_attention_heads,
        head_dim=args.head_dim,
    )
    if args.csv is not None:
        write_csv(
            args.csv,
            results,
            num_attention_heads=args.num_attention_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            block_m=args.block_m,
            block_n=args.block_n,
            num_warps=args.num_warps,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(f"\nraw CSV: {args.csv}")


if __name__ == "__main__":
    main()
