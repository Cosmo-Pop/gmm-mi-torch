# gmm-mi-torch

A pure-PyTorch port of [GMM-MI](https://github.com/dpiras/GMM-MI) ([Piras et al. 2023](https://dx.doi.org/10.1088/2632-2153/acc444)): estimation of mutual information — and related information-theoretic quantities — with bootstrap errorbars, using Gaussian mixture models. The port replaces GMM-MI's sklearn/loop machinery with a single batched, weighted EM that runs the whole procedure (initialisations × cross-validation folds, or all bootstrap realisations) as one tensor computation, on CPU or GPU, at float64 or float32.

## Quantities

Given data `X` of shape `(n_samples, n_variables)`, `estimate` fits **one joint GMM** (component count chosen by cross-validation) and computes the requested quantity by Monte Carlo over the fitted model, with every marginal obtained analytically from the joint fit:

- `'MI'` — mutual information I(X;Y) between two columns
- `'cMI'` — conditional mutual information I(X;Y|Z), conditioning on the remaining column(s)
- `'TC'` — total correlation of all columns
- `'KL_forward'` / `'KL_reverse'` — KL divergence between the two 1-D marginals
- `estimate_categorical(X, Y)` — MI between a continuous variable and a categorical one (a GMM per category), including the uniform-category-weight variant of Piras et al. (App. B)

All quantities are returned in nats. Errorbars come from a bootstrap: multinomial-count sample weights over `n_bootstrap` replicates, each warm-started from the selected fit, run as one batched EM.

## Installation

Optionally start from a fresh environment:

```bash
conda create -n gmm-mi-torch python=3.11
conda activate gmm-mi-torch
```

(or `python -m venv .venv && source .venv/bin/activate`). Then:

```bash
git clone https://github.com/Cosmo-Pop/gmm-mi-torch.git
cd gmm-mi-torch
pip install .          # or: pip install -e ".[test]"
```

The only runtime dependency is `torch` (numpy inputs are accepted; tests additionally need `numpy >= 2.0`, `scipy` and `pytest`, installed by the `test` extra). The plotting helpers in `gmm_mi_torch.plotting` need `numpy`, `matplotlib` and `scipy`, installed via the `plot` extra: `pip install ".[plot]"`.

## Quickstart

```python
import numpy as np
import torch
from gmm_mi_torch import estimate, estimate_categorical

rng = np.random.default_rng(0)
cov = np.array([[1.0, 0.6], [0.6, 1.0]])
X = rng.multivariate_normal([0, 0], cov, 10_000)

# defaults: CPU, float64, batched selection protocol
mi_mean, mi_std = estimate(X, 'MI')

# GPU, float32; replicate batches are sized automatically from free device memory
mi_mean, mi_std = estimate(X, 'MI', device='cuda', dtype=torch.float32)

# continuous-categorical MI
Y = rng.integers(0, 2, 10_000)
Z = rng.normal(2.0 * Y, 1.0)
mi_mean, mi_std = estimate_categorical(Z, Y, max_components=3)
```

The full hyperparameter set (selection protocols `'batched'`/`'sequential'`, `metric_method='valid'/'aic'/'bic'`, init types, `preprocess='gaussianise'`, `fixed_components_number`, chunking knobs, …) is documented in the `estimate` and `estimate_categorical` docstrings, and guidance on using `estimate` (component selection, standardisation, preprocessing, initialisation, memory and reproducibility) is in [docs/estimate_notes.md](docs/estimate_notes.md); parameter names follow GMM-MI where a counterpart exists (`reg_covar`, `threshold_fit`, `threshold_components`, `max_components`, …). `estimate(..., return_fit=True)` additionally returns the selected model for inspection and plotting (`gmm_mi_torch.plotting` has the fitted-ellipse, data-vs-model contour, MI-matrix heatmap and errorbar-validation helpers used in the walkthrough).

For a guided tour — including the common pitfalls (thresholds, initialisations, hard-boundary data) and how to diagnose them — see [notebooks/walkthrough_and_pitfalls.ipynb](notebooks/walkthrough_and_pitfalls.ipynb), adapted from Davide Piras's GMM-MI walkthrough.

## Performance

Benchmarked on an NVIDIA GB10 (DGX Spark) against the sklearn-based GMM-MI reference on a 3-component bivariate mixture (N=20 000, `max_components=8`, 50 bootstrap realisations); the gmm-mi-torch timings were measured at v0.2.1 with the default (`'auto'`) batch sizes. A distilled summary of the whole validation and benchmark campaign — correctness, fp32, calibration, performance, large-N scaling, known limitations — is in [docs/validation_summary.pdf](docs/validation_summary.pdf); full tables in [validation/gb10_report.md](validation/gb10_report.md) and the `validation/sweep_*` files:

| Config | model selection | bootstrap |
|---|---|---|
| GMM-MI (sklearn, CPU) | 15.4 s | 10.1 s |
| gmm-mi-torch, CPU fp64 | 2.0 s | 0.9 s |
| gmm-mi-torch, CUDA fp32 | **0.21 s** | **0.13 s** |

On real data at production scale (beta test, October 2026): a 75-pair MI grid between VAE latents and galaxy properties, N = 30,000, `max_components=20`, 100 bootstrap realisations per pair — GMM-MI on 16 CPU cores: 53 min wall-clock; gmm-mi-torch on one A100: 7.5 min at float32, **5.1 min at float64** (114–168× less, counting CPU-core-minutes against GPU-minutes). The two codes' MI values differ by 2σ or more only at MI ≈ 0, where both carry the estimator's positive finite-sample bias, and their bootstrap errorbars match to ~10%. On data-centre GPUs float64 is the faster dtype; the GB10-class caveat below is about rate-limited-fp64 hardware only.

The EM inner loop evaluates Gaussian log-densities and M-step moments as matrix products (GEMM, general matrix multiply) against per-sample quadratic features — a GMM log-density is a quadratic form in the sample, so scoring N samples against K components is one matrix multiplication, and the M-step moments are the transposed one; see [Machlica, Vaněk & Zajíc (2011)](https://ieeexplore.ieee.org/document/6118944) for the formulation. The numerical safeguards follow [Chan, Golub & LeVeque (1983)](https://www.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115); they are described in [docs/design_notes.md](docs/design_notes.md) ("GEMM-formulated EM kernels").

A second implementation of the EM step, which subtracts each component's mean from the samples before evaluating the quadratic form, is selected with `GMM_MI_TORCH_EM_KERNEL=legacy`; the warnings and the design notes call it the legacy kernels. The default kernels are roughly 3× faster per EM iteration on GPU and 2× on CPU (measured at v0.2.0), and at float64 the two give the same results up to the order of floating-point summation. Inputs are standardised internally (estimates are invariant; fitted models are returned in original units), which also removes float32 factorisation failures on unstandardised data.

### Replicate batches and reproducibility

- The replicate batches `em_chunk` and `mc_chunk` default to `'auto'`. On CPU `'auto'` is 8 replicates per batch.
- On CUDA each stage's batch is sized to fill half of the free device memory (torch's cached blocks included) from per-stage peak-memory models measured on the GB10 (`validation/bench_mem_models.py`; the models are tabulated in [docs/design_notes.md](docs/design_notes.md)).
- The batch is capped by `torch.cuda.set_per_process_memory_fraction` when one is set. There is no minimum batch size.
- A CUDA out-of-memory error halves the batch and retries, with a `CUDA out of memory; retrying with chunk=...` RuntimeWarning.
- The Monte-Carlo samples are independent of `mc_chunk` by construction: each bootstrap replicate draws them from its own seeded stream, so out-of-memory retries cannot shift them either.
- Results are independent of `em_chunk` and `mc_chunk` bitwise on CPU at float64; at float32 on CPU also bitwise except `em_chunk=1` with a one-component model, where the M-step's single-row tail product takes a different BLAS path (3e-8 in a measured MI). On CUDA either chunk changes the matmul summation order, so same-seed runs agree only to floating-point rounding unless both are set to integers and no out-of-memory retry occurs (the retry halves the batch whatever the setting, so a run's batch can depend on the memory free at the time).
- At float32 that rounding can change the selected k on data where two values of k score almost equally (k=5 vs 6, MI 0.4741 vs 0.4745 against an errorbar of 0.008 in a measured example); at float64 the differences stay at the level set by the order of summation (1e-14 to 1e-12 in the fitted means in two measured cases). A CPU run is the bit-reproducible setting (with that one float32 exception).

### Large N and the choice of dtype

Sample-dimension chunking (`n_chunk`) extends the reachable scale to N=10⁸, d=8 (total correlation in 23 min / 77.5 GiB at fp32 on 128 GB unified memory, measured at v0.2.x; v0.5.0 removed the two allocations that set that run's peak, the construction of the bootstrap counts and the repeated CV masks, and the run has not been repeated at N=10⁸ since).

On GPUs with rate-limited fp64 throughput (e.g. GB10), CUDA fp64 is slower than CPU fp64 — use fp32 on such devices (validated: the full test suite passes at fp32). On data-centre GPUs with full-rate fp64 (A100/H100/GH200 class), float64 costs only ~1.4x float32 end-to-end and avoids all reduced-precision caveats — prefer it there, especially for near-zero MI or spike-like marginals.

### Many pairs, many GPUs

A single `estimate()` call already fills one GPU, so on one device the fastest way through a grid of variable pairs is simply a loop. On a node with several GPUs, run one worker process per GPU and give each worker its own device:

```python
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import torch


def init_worker(counter, visible):
    # runs in each child before any CUDA use: pin this worker to one GPU.
    # Index INTO the inherited CUDA_VISIBLE_DEVICES mask — writing absolute
    # indices would escape a scheduler's allocation (e.g. under Slurm with
    # CUDA_VISIBLE_DEVICES=2,3, workers must pin to 2 and 3, not 0 and 1).
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[idx % len(visible)]


def one_pair(task):
    from gmm_mi_torch import estimate
    (i, j), X = task
    mi, sigma = estimate(X, "MI", device="cuda", dtype=torch.float64)
    return i, j, mi, sigma


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()       # GPUs visible to this job
    if n_gpus == 0:
        raise SystemExit("no CUDA device is visible")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = mask.split(",") if mask else [str(i) for i in range(n_gpus)]
    X = load_my_data()                     # (n_samples, n_variables) array
    pairs = [(i, j) for i in range(X.shape[1]) for j in range(i + 1, X.shape[1])]
    tasks = [((i, j), X[:, [i, j]]) for i, j in pairs]
    ctx = mp.get_context("spawn")   # fork is unsafe once CUDA is involved
    counter = ctx.Value("i", 0)
    with ProcessPoolExecutor(max_workers=n_gpus, mp_context=ctx,
                             initializer=init_worker,
                             initargs=(counter, visible)) as pool:
        for i, j, mi, sigma in pool.map(one_pair, tasks):
            print(i, j, mi, sigma)
```

This is also available as a runnable file: [`examples/mi_grid_multigpu.py`](examples/mi_grid_multigpu.py).

- It must be run as a script (`python examples/mi_grid_multigpu.py`): the `spawn` start method re-imports the file in each worker, so pasting the code into a notebook or an interactive Python/IPython session fails with `AttributeError: Can't get attribute ... on <module '__main__'>` followed by `BrokenProcessPool`.
- The `spawn` start method and the environment variable set in the initializer matter — CUDA contexts do not survive `fork`, and a worker must be pinned to its GPU before the first CUDA call.
- On one GPU this pool degenerates to a single worker and buys nothing over a plain loop; leave `max_workers` at the GPU count, never higher (an `estimate()` call already fills its GPU, so extra workers would only contend for the same devices).
- Use processes, not threads: the float32 precision setting each call makes is process-wide (see Known limitations).

## Documentation

The API reference in `docs/api/`, one page per module, is generated from the docstrings
(`python docs/make_api_docs.py`, which needs the `docs` extra):

- [docs/api/estimator.md](docs/api/estimator.md): `estimate`, `estimate_categorical`, `FittedGMM`
- [docs/api/em.md](docs/api/em.md): `em_fit`, `val_ll`
- [docs/api/selection.md](docs/api/selection.md): `select_batched`, `select_sequential`
- [docs/api/gaussian.md](docs/api/gaussian.md): Gaussian-mixture helpers (`gmm_score`, `gmm_sample`, `marginal_score`, ...)
- [docs/api/plotting.md](docs/api/plotting.md): plotting helpers

[docs/estimate_notes.md](docs/estimate_notes.md) gives guidance on using `estimate` (component
selection, standardisation, preprocessing, initialisation, memory and reproducibility), and
[docs/design_notes.md](docs/design_notes.md) explains why the code is shaped as it is, with the
measurements behind its constants.

## Known limitations

Mechanisms and measurements for each are in [docs/design_notes.md](docs/design_notes.md).

- **Finite-sample bias.** GMM MI estimates carry a positive finite-sample bias that the bootstrap errorbar does not include; below 100 samples (per dataset, or per category) the estimator warns.
- **Heavily tied or zero-inflated data.** No tie convention makes the GMM fit of heavily tied data reliable; `preprocess='none'` is the more reliable choice there (under `'gaussianise'`, random tie-ranks bias the estimate low when ties are shared between columns).
- **Bootstrap resample on CUDA.** For N ≤ 2^24 torch's CUDA multinomial has float32 resolution, so each sample's resampling weight carries a fixed factor, the same in every replicate, whose standard deviation about 1 is 0.02 at N=1e6, rising to ~0.24 near N=1e7. The factor does not depend systematically on a sample's position in the array, and its effect is a fixed shift of the bootstrap mean of order that standard deviation times one standard error; the CPU resample has no such factor. Above N=2^24 the index draw is a float64 uniform on both devices (bias below 3e-8 at N=1e8, with no dependence on position in the array).
- **CUDA reproducibility.** Same-seed CUDA runs agree only to floating-point rounding unless `em_chunk` and `mc_chunk` are set to integers and no out-of-memory retry occurs; at float32 the selected k can change on data where two values of k score almost equally (see above).
- **Per-replicate seeds on CPU.** CPU generators keep 32 bits of each per-replicate seed, so two replicates of one call share a stream with probability R(R−1)/2^33 (1.7e-6 at the default 120 replicates).
- **Threads.** For the duration of each call the package sets torch's float32 matrix-multiplication precision to full IEEE precision (TF32 off) and restores the caller's setting afterwards. That setting is process-wide, not per thread: run parallel estimates in processes, not threads.
- **Categorical errorbars.** `estimate_categorical` holds the category proportions fixed across bootstrap replicates, so its errorbar excludes their multinomial sampling uncertainty (negligible except at small N with extreme imbalance).
- **torch 2.13 without the Python development headers.** torch 2.13.0 evaluates GPU matrix products of the form (n×1)·(1×m), which include every 1×1 product, with a kernel that it compiles at first use, and the compilation needs `Python.h`. Where the header is missing, fits of one-dimensional data on CUDA, which include every `estimate_categorical` call, fail inside torch with a `subprocess.CalledProcessError`. Install the Python development headers (for example the `python3-dev` package), or set `TORCH_DISABLE_NATIVE_JIT=1` before importing torch. MI, KL, cMI and TC fits of two or three columns were unaffected in tests. Observed with torch 2.13.0+cu130 on the GB10; torch 2.11.0 is unaffected.

## Tests

```bash
pip install -e ".[test]"
pytest                                                    # CPU, float64
GMM_MI_DEVICE=cuda GMM_MI_DTYPE=float32 GMM_MI_EM_CHUNK=1000 pytest   # GPU, float32
```

The suite checks analytic Gaussian targets, Monte-Carlo/quadrature targets for multimodal and non-Gaussian cases, chunking parity, categorical MI, and error paths. `tests/conftest.py` routes the whole suite to a device/dtype via the `GMM_MI_*` environment variables; tests that call internals directly ignore the variables (most run on CPU at float64, while a few tests of numerical precision build float32 tensors, and the test of the float32 precision setting builds CUDA ones).

- Two tests are gated: the bootstrap-errorbar coverage calibration over 40 independent datasets (slow; runs under `GMM_MI_OFFLINE=1`) and the full-strength float32 spike-pair regression (runs when `GMM_MI_DEVICE=cuda` or `GMM_MI_OFFLINE=1`); a default CPU run skips both.
- Nine further tests need a CUDA device and skip on a CPU-only host, so a green CPU-only run has not exercised them: the measured-peak pins of the memory passes (`test_init_draw_is_chunked_on_device`, `test_moment_gemm_peak_on_device`, `test_categorical_mc_peak_independent_of_categories`, `test_streamed_path_holds_no_full_weight_copies`, `test_val_ll_streams_rerouted_replicates`'s peak half, `test_streamed_init_peak_under_model`), the float32 precision setting (`test_full_precision_tf32_gemm_result_cuda`), the default-device test (`test_cv_masks_survive_default_device`) and the error raised when not even one replicate's initialisation fits in memory (`test_init_hopeless_case_raises_readably`).

For most readers, [docs/validation_summary.pdf](docs/validation_summary.pdf) is the right entry point to the validation results. The [validation/](validation/) directory holds the full working report from the port's GB10 acceptance run and the benchmark scripts behind it ([validation/README.md](validation/README.md) explains its contents and provenance).

## Acknowledgements

Benedict Van Den Bussche beta-tested the port extensively on production-scale astrophysical MI grids against the reference implementation; the chunk auto-sizing, the float32 robustness fixes and the validation numbers quoted above all trace back to his reports. Davide Piras reviewed the repository and road-tested the examples.

## License and attribution

This is a port of [GMM-MI](https://github.com/dpiras/GMM-MI) by Davide Piras, Hiranya V. Peiris, Andrew Pontzen, Luisa Lucie-Smith, Ningyuan Guo and Brian Nord, distributed under the GPL-3.0 ([LICENSE](LICENSE)). Upstream GMM-MI additionally carries a non-commercial use condition; this repository is distributed without it, in agreement with the upstream author.

The walkthrough notebook and the plotting utilities (`gmm_mi_torch/plotting.py`) are forked from GMM-MI originals by Davide Piras, imported into this repository together with their upstream commit history — his authorship of that material is recorded in the git log — and then adapted to the `gmm-mi-torch` API here.

If you use this code, please cite the GMM-MI paper:

```bibtex
@article{Piras23,
     author = {Davide Piras and Hiranya V Peiris and Andrew Pontzen and
               Luisa Lucie-Smith and Ningyuan Guo and Brian Nord},
     title = {A robust estimator of mutual information for deep learning interpretability},
     journal = {Machine Learning: Science and Technology},
     doi = {10.1088/2632-2153/acc444},
     url = {https://dx.doi.org/10.1088/2632-2153/acc444},
     year = {2023},
     month = {apr},
     publisher = {IOP Publishing},
     volume = {4},
     number = {2},
     pages = {025006}
}
```
