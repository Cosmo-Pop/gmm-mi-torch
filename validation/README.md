# Validation artifacts (GB10 acceptance run)

Verbatim artifacts from the port's validation and benchmarking on an NVIDIA GB10
(DGX Spark, 128 GB unified memory, torch 2.11.0+cu128), July 2026.

At the time these were produced the module was a flat file named
`torch_gmm_mi.py` (its contents now live in `gmm_mi_torch/estimator.py`) and
the test suite was `test_torch_gmm_mi.py` (now `tests/test_gmm_mi_torch.py`,
identical up to the import rename). File names inside the artifacts reflect
the old layout. The bench scripts have since been repointed at the package
(`gmm_mi_torch.estimator`) and updated for v0.3.0's `_chol` signature change
(it returns `(chol, cov)` since then; between v0.3.0 and v0.3.1 all of them
except `bench.py` crashed on the old single-tensor contract), so they run
against the current code; the numbers recorded in the reports remain those
of the versions stated there.

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

- `bench_mem_models.py` — the memory-model tooling since v0.5.0 (hardening pass,
  2026-10-21): measured peak / model for every auto-sized stage (single-pass and
  streamed GEMM EM, rerouted EM, val_ll with and without rerouted replicates, the
  init stage single-pass and streamed, the MC stage) at fp32 and fp64; source of
  the per-replicate models in `estimator.py` (GEMM EM max(K+6, 2K+3) + 3Kq/512
  units of N_eff × itemsize, legacy (4d+4)K + 4, init (4d+3)K + 12, MC
  max((3d+2)K + d + 7, d² + 4d + 6), all × 1.03). Every row must stay at or
  below 1.0.
- `bench_mem.py` — peak CUDA memory per replicate for one EM iteration and one
  MC chunk; source of the ORIGINAL v0.1.1 auto-chunk models (EM: N×K×itemsize×(4d+3),
  MC: mc_samples×K×itemsize×(3d+1), measured to within 2% for d=1..8, K=5..20,
  fp32/fp64), superseded by the models above.
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
- `sweep_gh200.{json,md}` — `bench_sweep.py` output on an Isambard-AI GH200 120GB (132 SMs, torch 2.11.0+cu128, 2026-08-21), for comparison with `sweep_gb10*`.
- `sweep_gb10_v020.{json,md}` — `bench_sweep.py` on the same GB10 at v0.2.0 (GEMM EM
  kernels): hard pair 13.5 s fp32 / 91.6 s fp64 at `em_chunk='auto'` (v0.1.1 kernels:
  46.1 s / 1330 s); EM-step floor 0.20 ms fp32 / 1.14 ms fp64 per replicate-iteration.
- `bench_gemm_step.py` — GEMM-vs-legacy EM-step parity at float64 and the per-step
  timing behind the v0.2.0 numbers (206 -> 61.6 ms at Ra=300, N=20k, K=20, d=2,
  fp32 on GB10).
- `sweep_gh200_v020.{json,md}` — `bench_sweep.py` on the same Isambard-AI GH200 at v0.2.0 (GEMM EM kernels), for comparison with `sweep_gh200` (v0.1.1) and `sweep_gb10_v020`.
