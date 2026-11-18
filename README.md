# gmm-mi-torch
[![DOI](https://img.shields.io/badge/DOI-10.1088/2632--2153/acc444-%231682D4)](https://doi.org/10.1088/2632-2153/acc444)

A pure-PyTorch port of [GMM-MI](https://github.com/dpiras/GMM-MI) ([Piras et al. 2023](https://dx.doi.org/10.1088/2632-2153/acc444)): mutual information and related information-theoretic quantities, with bootstrap errorbars, from Gaussian mixture models. The port replaces GMM-MI's sklearn loops with a single batched, weighted EM that runs the whole procedure (initialisations × cross-validation folds, or all bootstrap realisations) as one tensor computation, on CPU or GPU, at float64 or float32.

If you use this code, please cite Piras et al. (2023); the BibTeX entry is at the end of this page.

## Quantities

Given data `X` of shape `(n_samples, n_variables)`, `estimate` fits one joint GMM (component count chosen by cross-validation) and computes the requested quantity by Monte Carlo over the fitted model, with every marginal obtained analytically from the joint fit:

- `'MI'`: mutual information I(X;Y) between two columns
- `'cMI'`: conditional mutual information I(X;Y|Z), conditioning on the remaining column(s)
- `'TC'`: total correlation of all columns
- `'KL_forward'` / `'KL_reverse'`: KL divergence between the two 1-D marginals
- `estimate_categorical(X, Y)`: MI between a continuous variable and a categorical one (a GMM per category)

All quantities are returned in nats. The errorbar is the standard deviation of the estimate over `n_bootstrap` bootstrap resamples of the data. A resample is represented by a weight per sample (the number of times that sample was drawn), not by a copy of the data; each refit starts from the selected model, and all of them run together as one batched EM.

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
pip install .
```

The only runtime dependency is `torch`. `numpy` arrays are accepted as input, but `numpy` is not installed with the package; the Quickstart below needs it, and so does `estimate_categorical` when the category labels are strings (`pip install numpy`). The plotting helpers need `numpy`, `matplotlib` and `scipy` (`pip install ".[plot]"`), and the tests need `pytest`, `numpy >= 2.0` and `scipy` (`pip install -e ".[test]"`).

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

- The full hyperparameter set (selection protocols `'batched'`/`'sequential'`, `metric_method='valid'/'aic'/'bic'`, init types, `preprocess='gaussianise'`, `fixed_components_number`, chunking knobs, …) is documented in the `estimate` and `estimate_categorical` docstrings.
- Guidance on using `estimate` (component selection, standardisation, preprocessing, initialisation, memory and reproducibility) is in [docs/estimate_notes.md](docs/estimate_notes.md).
- Parameter names follow GMM-MI where a counterpart exists (`reg_covar`, `threshold_fit`, `threshold_components`, `max_components`, …).
- `estimate(..., return_fit=True)` additionally returns the selected model for inspection and plotting; `gmm_mi_torch.plotting` has the fitted-ellipse, data-vs-model contour, MI-matrix heatmap and errorbar-validation helpers used in the walkthrough.

For a guided tour — including the common pitfalls (thresholds, initialisations, hard-boundary data) and how to diagnose them — see [notebooks/walkthrough_and_pitfalls.ipynb](notebooks/walkthrough_and_pitfalls.ipynb), adapted from Davide Piras's GMM-MI walkthrough.

## Performance

On a 3-component bivariate mixture (N=20 000, `max_components=8`, 50 bootstrap realisations), on an NVIDIA GB10, measured at v0.2.1:

| Config | model selection | bootstrap |
|---|---|---|
| GMM-MI (sklearn, CPU) | 15.4 s | 10.1 s |
| gmm-mi-torch, CPU float64 | 2.0 s | 0.9 s |
| gmm-mi-torch, CUDA float32 | 0.21 s | 0.13 s |

On real data at production scale (beta test, October 2026): the MI of each of 75 pairs of variables, the latent variables of a variational autoencoder against galaxy properties (as in [Van den Bussche et al. 2026](https://arxiv.org/abs/2606.11308)), N = 30,000, `max_components=20`, 100 bootstrap realisations per pair — GMM-MI on 16 CPU cores: 53 min wall-clock; gmm-mi-torch on one A100: 7.5 min at float32, **5.1 min at float64** (114–168× less, counting CPU-core-minutes against GPU-minutes). The two codes' MI values differ by 2σ or more only at MI ≈ 0, where both carry the estimator's positive finite-sample bias, and their bootstrap errorbars match to ~10%. On data-centre GPUs float64 is the faster dtype; the GB10-class caveat below is about hardware with slow float64 only.

On GPUs whose float64 arithmetic runs much slower than their float32 (the GB10 and consumer cards), CUDA float64 is slower than CPU float64 — use float32 there (the full test suite passes at float32). On data-centre GPUs, where float64 runs at full speed (A100 and H100, including the GH200), float64 costs about the same as float32, or less, and avoids all reduced-precision caveats — prefer it there, especially for near-zero MI or spike-like marginals.

The EM inner loop evaluates Gaussian log-densities and M-step moments as matrix products (GEMM, general matrix multiply) against per-sample quadratic features (each sample's coordinates and their pairwise products) — a GMM log-density is a quadratic form in the sample, so scoring N samples against K components is one matrix multiplication, and the M-step moments are the transposed one; see [Machlica, Vaněk & Zajíc (2011)](https://ieeexplore.ieee.org/document/6118944) for the formulation. The numerical safeguards follow [Chan, Golub & LeVeque (1983)](https://www.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115); they are described in [docs/design_notes.md](docs/design_notes.md) ("GEMM-formulated EM kernels").

A second implementation of the EM step, which subtracts each component's mean from the samples before evaluating the quadratic form, is selected with `GMM_MI_TORCH_EM_KERNEL=legacy`; the warnings and the design notes call it the legacy kernels. The default kernels are roughly 3× faster per EM iteration on GPU and 2× on CPU (measured at v0.2.0), and at float64 the two give the same results up to the order of floating-point summation. Inputs are standardised internally (estimates are invariant; fitted models are returned in original units), which also removes float32 factorisation failures on unstandardised data.

Batch sizes, reproducibility across devices and data of up to N=10⁸ samples are covered in [docs/performance_notes.md](docs/performance_notes.md).

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
(`python docs/make_api_docs.py`; `pip install ".[docs]"` installs what it needs):

- [docs/api/estimator.md](docs/api/estimator.md): `estimate`, `estimate_categorical`, `FittedGMM`
- [docs/api/em.md](docs/api/em.md): `em_fit`, `val_ll`
- [docs/api/selection.md](docs/api/selection.md): `select_batched`, `select_sequential`
- [docs/api/gaussian.md](docs/api/gaussian.md): Gaussian-mixture helpers (`gmm_score`, `gmm_sample`, `marginal_score`, ...)
- [docs/api/plotting.md](docs/api/plotting.md): plotting helpers

[docs/estimate_notes.md](docs/estimate_notes.md) gives guidance on using `estimate` (component
selection, standardisation, preprocessing, initialisation, memory and reproducibility), and
[docs/design_notes.md](docs/design_notes.md) explains why the code is shaped as it is, with the
measurements behind its constants.
[docs/performance_notes.md](docs/performance_notes.md) covers batch sizes and reproducibility, and large N.

## Known limitations

Mechanisms and measurements for each are in [docs/design_notes.md](docs/design_notes.md).

- MI estimates from a GMM fit carry a positive finite-sample bias that the bootstrap errorbar does not include; below 100 samples the estimator warns.
- Data with many repeated values, or with a large share of exact zeros, are not fitted reliably by a GMM, however `preprocess='gaussianise'` ranks the repeats; `preprocess='none'` is the more reliable choice there.
- On CUDA, two runs with the same seed agree only to floating-point rounding, unless `em_chunk` and `mc_chunk` are set to integers and neither run meets a CUDA out-of-memory error (after one, the batch is halved and the stage repeated). A CPU run reproduces bit for bit, with one float32 exception described in [docs/estimate_notes.md](docs/estimate_notes.md).
- On CUDA with N ≤ 2^24, the bootstrap does not draw every sample with exactly equal probability: each sample's probability differs from 1/N by a fixed factor, with a standard deviation of 0.02 at N=1e6, rising to about 0.24 near N=1e7. The CPU resample is uniform.
- Run parallel estimates in processes, not threads: each call sets torch's float32 matrix-multiplication precision for the whole process while it runs.
- `estimate_categorical` holds the category proportions fixed across bootstrap replicates, so its errorbar excludes their sampling uncertainty.
- **torch 2.13 without the Python development headers.** torch 2.13.0 evaluates GPU matrix products of the form (n×1)·(1×m), which include every 1×1 product, with a kernel that it compiles at first use, and the compilation needs `Python.h`. Where the header is missing, fits of one-dimensional data on CUDA, which include every `estimate_categorical` call, fail inside torch with a `subprocess.CalledProcessError`. Install the Python development headers (for example the `python3-dev` package), or set `TORCH_DISABLE_NATIVE_JIT=1` before importing torch. MI, KL, cMI and TC fits of two or three columns were unaffected in tests with the default kernels. Observed with torch 2.13.0+cu130 on the GB10; torch 2.11.0 is unaffected.

## Tests

```bash
pip install -e ".[test]"
pytest                                                    # CPU, float64
GMM_MI_DEVICE=cuda GMM_MI_DTYPE=float32 GMM_MI_EM_CHUNK=1000 pytest   # GPU, float32
```

A CPU-only run skips two tests that run only when an environment variable is set, and nine that need a CUDA device; [tests/README.md](tests/README.md) lists them.

[docs/validation_summary.pdf](docs/validation_summary.pdf) is a three-page summary of the validation and benchmark results. The [validation/](validation/) directory holds the full working report from the port's GB10 acceptance run and the benchmark scripts behind it ([validation/README.md](validation/README.md) explains its contents and provenance).

## Acknowledgements

Benedict Van Den Bussche beta-tested the port extensively on production-scale astrophysical MI grids against the reference implementation; the chunk auto-sizing, the float32 robustness fixes and the validation numbers quoted above all trace back to his reports. Davide Piras reviewed the repository and road-tested the examples.

## License and attribution

This is a port of [GMM-MI](https://github.com/dpiras/GMM-MI) by Davide Piras, Hiranya V. Peiris, Andrew Pontzen, Luisa Lucie-Smith, Ningyuan Guo and Brian Nord, distributed under the GPL-3.0 ([LICENSE](LICENSE)). Upstream GMM-MI additionally carries a non-commercial use condition; this repository is distributed without it, in agreement with the upstream author.

The walkthrough notebook and the plotting utilities (`gmm_mi_torch/plotting.py`) are forked from GMM-MI originals by Davide Piras, imported into this repository together with their upstream commit history — his authorship of that material is recorded in the git log — and then adapted to the `gmm-mi-torch` API here.

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
