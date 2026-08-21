# Validation artifacts (GB10 acceptance run)

Verbatim artifacts from the port's validation and benchmarking on an NVIDIA GB10
(DGX Spark, 128 GB unified memory, torch 2.11.0+cu128), July 2026. Kept in the
repository for team review while it is private; this directory can be pruned or
moved before a public release.

At the time these were produced the module was a flat file named
`torch_gmm_mi.py` (now `gmm_mi_torch/__init__.py`, contents identical) and the
test suite was `test_torch_gmm_mi.py` (now `tests/test_gmm_mi_torch.py`,
identical up to the import rename). File names inside the artifacts reflect the
old layout; the bench scripts likewise do `import torch_gmm_mi` and would need
that one-line rename to run against the installed package.

- `gb10_report.md` — the full report, Parts 1–6: fp32/CUDA validation of the
  original port, benchmark and scale tables, the extensions
  (init types, gaussianise preprocessing, AIC/BIC selection, marginal KL,
  categorical MI, chunking), sample-dimension chunking to N=10⁸, the polished
  module's parameter renames, and the extended 24-test suite.
- `briefs/` — the task briefs the work responded to.
- `changes.diff` — task-scoped diff of the final Part-6 changes (extended test
  suite + one module seam) against the shipped Part-5 baseline.
- `bench.py`, `bench_scale.py`, `bench_nchunk.py` — benchmark scripts behind
  the report's timing/memory tables.
- `time_cpu.txt`, `time_cuda.txt` — `/usr/bin/time -v` logs of the final full
  suite runs (CPU float64 and CUDA float32).

## Throughput scoping (August 2026)

Scripts behind the "parallelise MI over pairs" scoping and the `em_chunk='auto'`
default (v0.1.1), all run on the same GB10 with `PYTHONPATH=<repo> python <script>`:

- `bench_mem.py` — peak CUDA memory per replicate for one EM iteration and one
  MC chunk; source of the memory model in `estimator.py` (EM: N×K×itemsize×(4d+3),
  MC: mc_samples×K×itemsize×(3d+1), measured to within 2% for d=1..8, K=5..20, fp32/fp64).
- `bench_single.py` — one N=20k pair at the beta tester's settings: phase breakdown
  (selection / bootstrap / MC) and wall time vs `em_chunk`.
- `bench_iters.py` — per-iteration trace of the active replicate count (straggler
  tail) and per-replicate EM cost vs batch size.
- `bench_smallN.py` — the batch-size saturation curve at N=2k, 5k, 20k.
- `bench_prof.py` — the pair-stacking proxy: 4×300 replicates serially vs 1200 in one
  batch (1.02× on GB10).
- `bench_gemm.py` — GEMM-formulated E/M-step prototypes vs the current kernels, and
  the sequential-protocol timing.
- `bench_bw.py` — elementwise vs `logsumexp` bandwidth calibration; MI under both
  selection protocols.
- `bench_accept_A.py` — acceptance run for the `'auto'` chunk default (72.8 s → 46.1 s).
- `bench_sweep.py` — self-contained hardware sweep (synthetic hard pair): `estimate()` wall
  time vs `em_chunk` at float32 and float64, plus the EM-step saturation curve; writes
  `<out>.json` / `<out>.md`. Intended for running on other GPUs (A100, GH200) and comparing
  against the GB10 numbers above. `--quick` is a one-minute smoke test.
- `sweep_gh200.{json,md}` — `bench_sweep.py` output on an Isambard-AI GH200 120GB (132 SMs, torch 2.11.0+cu128, 2026-08-22), for comparison with `sweep_gb10*`.
