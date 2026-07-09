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

```bash
git clone git@github.com:Cosmo-pop/gmm-mi-torch.git
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

# GPU, float32, chunks tuned for throughput
mi_mean, mi_std = estimate(X, 'MI', device='cuda', dtype=torch.float32,
                           em_chunk=48, mc_chunk=50)

# continuous-categorical MI
Y = rng.integers(0, 2, 10_000)
Z = rng.normal(2.0 * Y, 1.0)
mi_mean, mi_std = estimate_categorical(Z, Y, max_components=3)
```

The full hyperparameter set (selection protocols `'batched'`/`'sequential'`, `metric_method='valid'/'aic'/'bic'`, init types, `preprocess='gaussianise'`, `fixed_components_number`, chunking knobs, …) is documented in the `estimate` and `estimate_categorical` docstrings; parameter names follow GMM-MI where a counterpart exists (`reg_covar`, `threshold_fit`, `threshold_components`, `max_components`, …).

## Performance

Benchmarked on an NVIDIA GB10 (DGX Spark) against the sklearn-based GMM-MI reference on a 3-component bivariate mixture (N=20 000, `max_components=8`, 50 bootstrap realisations); full tables in [validation/gb10_report.md](validation/gb10_report.md):

| Config | model selection | bootstrap |
|---|---|---|
| GMM-MI (sklearn, CPU) | 15.4 s | 10.1 s |
| gmm-mi-torch, CPU fp64 | 11.5 s | 1.9 s |
| gmm-mi-torch, CUDA fp32 (tuned chunks) | **0.90 s** | **0.17 s** |

Sample-dimension chunking (`n_chunk`) extends the reachable scale to N=10⁸, d=8 (total correlation in 23 min / 77.5 GiB at fp32 on 128 GB unified memory). One hardware caveat: on GPUs with rate-limited fp64 throughput (e.g. GB10), CUDA fp64 is slower than CPU fp64 — use fp32 on such devices (validated: the full test suite passes at fp32).

## Tests

```bash
pip install -e .[test]
pytest                                                    # CPU, float64
GMM_MI_DEVICE=cuda GMM_MI_DTYPE=float32 GMM_MI_EM_CHUNK=1000 pytest   # GPU, float32
```

The suite (24 tests) checks analytic Gaussian targets, Monte-Carlo/quadrature targets for multimodal and non-Gaussian cases, chunking parity, categorical MI, and error paths; `tests/conftest.py` routes the whole suite to a device/dtype via the `GMM_MI_*` environment variables. One additional statistical test — bootstrap-errorbar coverage calibration over 40 independent datasets — is slow and therefore gated behind `GMM_MI_OFFLINE=1`.

The [validation/](validation/) directory contains the full validation and benchmark report from the port's GB10 acceptance run ([validation/README.md](validation/README.md) explains its contents and provenance), kept in the repository for team review while it is private.

## License and attribution

This is a port of [GMM-MI](https://github.com/dpiras/GMM-MI) by Davide Piras, Hiranya V. Peiris, Andrew Pontzen, Luisa Lucie-Smith, Ningyuan Guo and Brian Nord, and mirrors its licensing: GPL-3.0 ([LICENSE](LICENSE)) with upstream's non-commercial use condition ([LICENSE_EXT.txt](LICENSE_EXT.txt)). Review before making this repository public.

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
