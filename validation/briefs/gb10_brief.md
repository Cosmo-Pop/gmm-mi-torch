# GB10 task: GPU benchmark and fp32 validation of torch_gmm_mi

Files: torch_gmm_mi.py (module), test_torch_gmm_mi.py (pytest suite, all 5 pass on CPU/fp64).

- Run the pytest suite unchanged on GB10 CPU (fp64) to confirm environment parity.
- Run the suite on CUDA with `device='cuda', dtype=torch.float64`; fix any residual device-placement errors — the module has been device-audited but not executed on a CUDA device.
- Rerun the suite at `dtype=torch.float32`; if any test fails, check Cholesky conditioning first (raise `reg` to 1e-5 before touching anything else) and report which quantity broke and by how many sigma.
- Benchmark on the 3-component-mixture workload (construction in the conversation transcript; N=20000, max_k=8, n_bootstrap=50): wall-clock for `select_batched` and for the bootstrap phase, at fp64 and fp32, CPU vs GPU. CPU sklearn reference: selection 15.4 s, bootstrap 10.1 s.
- Scale test: repeat at N=200000 and d=8 (8-dim correlated Gaussian, TC), reporting peak GPU memory; tune the `chunk` parameter in `em_fit` and the MC chunk (hardcoded 8 in `estimate`) — expose the latter as a parameter if it binds.
- Do not change the parsimony selection rule, the estimator integrands, or test tolerances; numerical-stability changes to EM internals are fine if all tests still pass.
- Deliverables: timing table, peak-memory numbers, fp32 pass/fail per test, diff of any changes.
