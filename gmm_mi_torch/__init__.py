"""gmm-mi-torch: mutual information with bootstrap errorbars from GMMs.

Pure-torch port of GMM-MI (Piras et al. 2023). The entry points live in
`gmm_mi_torch.estimator`, built on `constants`, `precision`, `chunking`,
`gaussian`, `em` and `selection` (design notes in docs/design_notes.md);
plotting helpers live in `gmm_mi_torch.plotting`.
"""
from .estimator import FittedGMM, estimate, estimate_categorical

__version__ = '0.11.2'   # keep in sync with pyproject.toml
__all__ = ['FittedGMM', 'estimate', 'estimate_categorical', '__version__']
