# WSL2 development environment

This is the daily Linux development environment for `einf` on the Windows host
`DESKTOP-B5DI30R`. The RTX 4090 host remains a separate validation and
performance-comparison machine; results from the two GPUs must not be combined
into one latency baseline.

## Host and GPU boundary

- Windows entry point: `Administrator@100.94.140.84`
- WSL distribution: Ubuntu-24.04, WSL2
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, driver 596.21, compute capability
  `(12, 0)`, 8151 MiB
- Daily workspace: `/home/administrator/work/einf` on the WSL ext4 filesystem
- RTX 4090 authoritative performance host: `zzx@10.126.126.2`; its repository
  and CUDA environment are independent and must not be overwritten from WSL.

Keep CUDA/C++ builds in WSL ext4 rather than `/mnt/c` or `/mnt/d`.

## Activate and verify

POSIX shells:

```bash
. "$HOME/work/activate-einf.sh"
einf-env-check
```

Fish:

```fish
source "$HOME/work/activate-einf.fish"
einf-env-check
```

The activation script sets `EINF_ROOT`, `PYTHONPATH`, the Python 3.12 virtual
environment, CUDA 13.2, `TORCH_EXTENSIONS_DIR`, and `MAX_JOBS=2`. The helper
`einf-env-check` prints the toolchain, GPU, PyTorch CUDA availability, BF16
support, and CuTe import status.

## Fixed toolchain

| component | value |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| system CUDA toolkit / nvcc | 13.2 (`/usr/local/cuda-13.2`) |
| CuTe DSL | `nvidia-cutlass-dsl` 4.7.1 |
| TVM FFI | `apache-tvm-ffi` 0.1.13.post3 |
| Ninja | 1.13.0 |
| Triton | 3.6.0 |
| CUTLASS headers | commit `c506e16788cb08416a4a57e11a9067beeee29420` |

The CUDA toolkit supplies headers, `nvcc`, and C++ extension link libraries.
The PyTorch wheels supply their user-space CUDA runtime. Do not install a Linux
NVIDIA driver inside WSL; `/usr/lib/wsl/lib` is provided by the Windows driver.

The WSL CUDA apt repository is configured by the installed CUDA keyring. The
minimal build packages and `cuda-libraries-dev-13-2` are sufficient for the
current PyTorch custom-op build; `cuda` and `cuda-drivers` meta-packages are
intentionally not part of this setup.

## Proxy

Windows Mihomo TUN exposes the WSL-reachable mixed HTTP/SOCKS endpoint
`http://127.0.0.1:10090`. The proxy configuration is stored in:

- `$HOME/.config/einf/proxy.env`
- `$HOME/.config/pip/pip.conf`
- `/etc/apt/apt.conf.d/80-einf-windows-proxy`

For explicit Python/curl operations in a shell, source `proxy.env`. For pip,
unset `ALL_PROXY`/`all_proxy` if a SOCKS setting is inherited and let pip use
its HTTP proxy configuration.

## Repository and tests

The WSL checkout was transferred as a separate worktree with the current dirty
state preserved. Do not reset or clean it to make a test pass. The ignored
`third_party/cutlass` checkout is sparse and pinned to the commit above.

Typical checks:

```bash
cd "$EINF_ROOT"
python -m pip check
python -m pytest -q --disable-warnings --maxfail=1
```

The current full suite includes CuTe tests. The known intentional educational
scaffold tests for `flash_attention_single_tile.py` remain `5 xfailed`; those
markers mean the rung is not implemented, not that the kernel is correct. The
current `flash_attention.py` test group is not green in this transferred dirty
state: its live source calls `_online_softmax_update` with five arguments while
the transferred `online_softmax_layout.py` helper accepts four. Resolve that
source/API mismatch in the user-owned DSL work before treating the full suite as
green.

Custom CUDA operators use the isolated cache and capped parallelism configured
by activation:

```bash
MAX_JOBS=2 python -m pytest -q --disable-warnings tests/test_torch_custom_ops.py
```

The first complete build on this host produced `89 passed` custom-op tests.
Control-plane/cache tests produced `30 passed`, and the non-FlashAttention suite
(excluding the known source mismatch) produced `181 passed, 5 xfailed`.

## CuTe TVM-FFI note

For direct Torch-tensor CuTe compilation with CuTe DSL 4.7.1, use the TVM-FFI
path explicitly, for example `cute.compile(..., options='--enable-tvm-ffi')`,
and keep `apache-tvm-ffi` installed. A minimal vector-add smoke has passed on
SM120.

## CUDA compatibility note

The RTX 5060/SM120 environment is for development and exploratory functional
checks. Compare its performance only within matched runs on this host. The
RTX 4090/SM89 CuTe benchmarks use CUDA 12.1 and remain the authoritative
performance results for that machine.
