# Nsight Compute helper scripts

These scripts run on the remote einf server from the repository root.

## Basic profile

```bash
scripts/profile_flash_attention_basic.sh
```

Defaults to `q_len=128`, `kv_len=32768`, `warmup=10`, and writes:

```text
/tmp/einf-flash-chunk-basic.ncu-rep
```

Override the shape and report path with environment variables:

```bash
Q_LEN=512 KV_LEN=512 \
REPORT=/tmp/einf-flash-prefill-512-basic \
scripts/profile_flash_attention_basic.sh
```

## Full profile

```bash
scripts/profile_flash_attention_full.sh
```

This collects occupancy, memory, instruction, and Warp-stall sections and
writes `/tmp/einf-flash-chunk-full.ncu-rep` by default. Full profiling is much
slower because Nsight Compute replays the Kernel for mutually exclusive metric
sets.

## Inspect a report

```bash
scripts/inspect_flash_attention_report.sh
scripts/inspect_flash_attention_report.sh \
  /tmp/einf-flash-chunk-basic.ncu-rep
```

For a CSV-like raw dump:

```bash
ncu \
  --import /tmp/einf-flash-chunk-full.ncu-rep \
  --page raw \
  --csv
```

The Extension is built with CUDA `-lineinfo`, so the Source page can correlate
reported instructions with CUDA source lines.
