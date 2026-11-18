# Tests

How to run the suite is in the [README](../README.md#tests).

The suite checks analytic Gaussian targets, Monte-Carlo/quadrature targets for multimodal and non-Gaussian cases, chunking parity, categorical MI, and error paths. `tests/conftest.py` routes the whole suite to a device/dtype via the `GMM_MI_*` environment variables; tests that call internals directly ignore the variables (most run on CPU at float64, while a few tests of numerical precision build float32 tensors, and the test of the float32 precision setting builds CUDA ones).

- Two tests are gated: the bootstrap-errorbar coverage calibration over 40 independent datasets (slow; runs under `GMM_MI_OFFLINE=1`) and the full-strength float32 spike-pair regression (runs when `GMM_MI_DEVICE=cuda` or `GMM_MI_OFFLINE=1`); a default CPU run skips both.
- Nine further tests need a CUDA device and skip on a CPU-only host, so a green CPU-only run has not exercised them: the measured-peak pins of the memory passes (`test_init_draw_is_chunked_on_device`, `test_moment_gemm_peak_on_device`, `test_categorical_mc_peak_independent_of_categories`, `test_streamed_path_holds_no_full_weight_copies`, `test_val_ll_streams_rerouted_replicates`'s peak half, `test_streamed_init_peak_under_model`), the float32 precision setting (`test_full_precision_tf32_gemm_result_cuda`), the default-device test (`test_cv_masks_survive_default_device`) and the error raised when not even one replicate's initialisation fits in memory (`test_init_hopeless_case_raises_readably`).
