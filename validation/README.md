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
