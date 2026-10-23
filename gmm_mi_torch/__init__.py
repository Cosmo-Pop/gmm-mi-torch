"""gmm-mi-torch: mutual information with bootstrap errorbars from GMMs.

Pure-torch port of GMM-MI (Piras et al. 2023). The estimator lives in
`gmm_mi_torch.estimator` (see its module docstring for the design notes);
plotting helpers live in `gmm_mi_torch.plotting`.
"""
from .estimator import FittedGMM, estimate, estimate_categorical

__version__ = '0.6.1'   # keep in sync with pyproject.toml
__all__ = ['FittedGMM', 'estimate', 'estimate_categorical', '__version__']
