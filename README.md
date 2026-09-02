# gmm-mi-torch

A pure-PyTorch port of [GMM-MI](https://github.com/dpiras/GMM-MI) ([Piras et al. 2023](https://dx.doi.org/10.1088/2632-2153/acc444)): estimation of mutual information — and related information-theoretic quantities — with bootstrap errorbars, using Gaussian mixture models. The port replaces GMM-MI's sklearn/loop machinery with a single batched, weighted EM that runs the whole procedure (initialisations × cross-validation folds, or all bootstrap realisations) as one tensor computation, on CPU or GPU, at float64 or float32.

## Quantities

Given data `X` of shape `(n_samples, n_variables)`, `estimate` fits **one joint GMM** (component count chosen by cross-validation) and computes the requested quantity by Monte Carlo over the fitted model, with every marginal obtained analytically from the joint fit:

- `'MI'` — mutual information I(X;Y) between two columns
- `'cMI'` — conditional mutual information I(X;Y|Z), conditioning on the remaining column(s)
- `'TC'` — total correlation of all columns
- `'KL_forward'` / `'KL_reverse'` — KL divergence between the two 1-D marginals
- `estimate_categorical(X, Y)` — MI between a continuous variable and a categorical one (a GMM per category), including the uniform-category-weight variant of Piras et al. (App. B)

Errorbars come from a bootstrap: multinomial-count sample weights over `n_bootstrap` replicates, each warm-started from the selected fit, run as one batched EM.

## Installation

Optionally start from a fresh environment:

```bash
conda create -n gmm-mi-torch python=3.11
conda activate gmm-mi-torch
```

(or `python -m venv .venv && source .venv/bin/activate`). Then:

```bash
git clone git@github.com:Cosmo-Pop/gmm-mi-torch.git
cd gmm-mi-torch
pip install .          # or: pip install -e .[test]
```

The only runtime dependency is `torch` (numpy inputs are accepted; tests additionally need `numpy` and `pytest`).

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

The full hyperparameter set (selection protocols `'batched'`/`'sequential'`, `metric_method='valid'/'aic'/'bic'`, init types, `preprocess='gaussianise'`, `fixed_components_number`, chunking knobs, …) is documented in the `estimate` and `estimate_categorical` docstrings; parameter names follow GMM-MI where a counterpart exists (`reg_covar`, `threshold_fit`, `threshold_components`, `max_components`, …). `estimate(..., return_fit=True)` additionally returns the selected model for inspection and plotting (`gmm_mi_torch.plotting` has the fitted-ellipse, data-vs-model contour, MI-matrix heatmap and errorbar-validation helpers used in the walkthrough).

For a guided tour — including the common pitfalls (thresholds, initialisations, hard-boundary data) and how to diagnose them — see [notebooks/walkthrough_and_pitfalls.ipynb](notebooks/walkthrough_and_pitfalls.ipynb), adapted from Davide Piras's GMM-MI walkthrough.

## Performance

Benchmarked on an NVIDIA GB10 (DGX Spark) against the sklearn-based GMM-MI reference on a 3-component bivariate mixture (N=20 000, `max_components=8`, 50 bootstrap realisations). A distilled summary of the whole validation and benchmark campaign — correctness, fp32, calibration, performance, large-N scaling, known limitations — is in [docs/validation_summary.pdf](docs/validation_summary.pdf); full tables in [validation/gb10_report.md](validation/gb10_report.md):

| Config | model selection | bootstrap |
|---|---|---|
| GMM-MI (sklearn, CPU) | 15.4 s | 10.1 s |
| gmm-mi-torch, CPU fp64 | 11.5 s | 1.9 s |
| gmm-mi-torch, CUDA fp32 (tuned chunks) | **0.90 s** | **0.17 s** |

Since v0.2.0 the EM inner loop evaluates Gaussian log-densities and M-step moments as matrix products against per-sample quadratic features — a GMM log-density is a quadratic form in the sample, so scoring N samples against K components is one matrix multiplication, and the M-step moments are the transposed one; see [Machlica, Vaněk & Zajíc (2011)](https://ieeexplore.ieee.org/document/6118944) for the formulation and the comments in `gmm_mi_torch/estimator.py` (kernel section) for the numerical safeguards, which follow [Chan, Golub & LeVeque (1983)](https://www.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115). This is several times faster than the previous broadcast kernels on both CPU and GPU at identical (float64 reduction-order) results; set `GMM_MI_TORCH_EM_KERNEL=legacy` to fall back to the previous kernels for one release. Inputs are standardised internally (estimates are invariant; fitted models are returned in original units), which also removes float32 factorisation failures on unstandardised data.

On CUDA the replicate batches (`em_chunk`, `mc_chunk`) default to `'auto'`: each stage's batch is sized to fill half of the free device memory from a measured peak-memory model, with a floor of 32 replicates (per-replicate cost is flat above that), and a CUDA out-of-memory error halves the batch and retries. Results are batch-size invariant, so the setting only trades memory against speed; pass an integer to pin it. On CPU `'auto'` keeps the previous default of 8.

### Many pairs, many GPUs

A single `estimate()` call already fills one GPU, so on one device the fastest way through a grid of variable pairs is simply a loop. On a node with several GPUs, run one worker process per GPU and give each worker its own device:

```python
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import torch

N_GPUS = max(torch.cuda.device_count(), 1)   # GPUs visible to this job


def init_worker(counter, n_gpus):
    # runs in each child before any CUDA use: pin this worker to one GPU
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx % n_gpus)


def one_pair(task):
    from gmm_mi_torch import estimate
    (i, j), X = task
    mi, sigma = estimate(X, "MI", device="cuda", dtype=torch.float64)
    return i, j, mi, sigma


if __name__ == "__main__":
    X = load_my_data()                     # (n_samples, n_variables) array
    pairs = [(i, j) for i in range(X.shape[1]) for j in range(i + 1, X.shape[1])]
    tasks = [((i, j), X[:, [i, j]]) for i, j in pairs]
    ctx = mp.get_context("spawn")   # fork is unsafe once CUDA is involved
    counter = ctx.Value("i", 0)
    with ProcessPoolExecutor(max_workers=N_GPUS, mp_context=ctx,
                             initializer=init_worker,
                             initargs=(counter, N_GPUS)) as pool:
        for i, j, mi, sigma in pool.map(one_pair, tasks):
            print(i, j, mi, sigma)
```

The `spawn` start method and the environment variable set in the initializer matter: CUDA contexts do not survive `fork`, and a worker must be pinned to its GPU before the first CUDA call. On one GPU this pool degenerates to a single worker and buys nothing over a plain loop — leave `max_workers` at the GPU count, never higher (a worker pinned to a GPU that does not exist crashes the pool). Process pools also do not work from inside a notebook (the workers cannot import functions defined there); keep this pattern in a script.

Sample-dimension chunking (`n_chunk`) extends the reachable scale to N=10⁸, d=8 (total correlation in 23 min / 77.5 GiB at fp32 on 128 GB unified memory). One hardware caveat: on GPUs with rate-limited fp64 throughput (e.g. GB10), CUDA fp64 is slower than CPU fp64 — use fp32 on such devices (validated: the full test suite passes at fp32). On data-centre GPUs with full-rate fp64 (A100/H100/GH200 class), float64 costs only ~1.4x float32 end-to-end and avoids all reduced-precision caveats — prefer it there, especially for near-zero MI or spike-like marginals.

## Tests

```bash
pip install -e .[test]
pytest                                                    # CPU, float64
GMM_MI_DEVICE=cuda GMM_MI_DTYPE=float32 GMM_MI_EM_CHUNK=1000 pytest   # GPU, float32
```

The suite (35 tests) checks analytic Gaussian targets, Monte-Carlo/quadrature targets for multimodal and non-Gaussian cases, chunking parity, categorical MI, and error paths; `tests/conftest.py` routes the whole suite to a device/dtype via the `GMM_MI_*` environment variables. One additional statistical test — bootstrap-errorbar coverage calibration over 40 independent datasets — is slow and therefore gated behind `GMM_MI_OFFLINE=1`.

For most readers, [docs/validation_summary.pdf](docs/validation_summary.pdf) is the right entry point to the validation results. The [validation/](validation/) directory additionally contains the full working report from the port's GB10 acceptance run ([validation/README.md](validation/README.md) explains its contents and provenance), kept in the repository for team review while it is private.

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
