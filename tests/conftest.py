"""Test-harness shim: route the suite to a device/dtype via env vars.

GMM_MI_DEVICE=cpu|cuda, GMM_MI_DTYPE=float64|float32, GMM_MI_REG=<float>,
GMM_MI_EM_CHUNK=<int> (defaults reproduce the original suite: cpu, float64,
module-default reg and chunks).  Wraps gmm_mi_torch.estimate and
gmm_mi_torch.estimate_categorical before the test module imports them.
"""

import os

import torch
import gmm_mi_torch

_DEVICE = os.environ.get("GMM_MI_DEVICE", "cpu")
_DTYPE = {"float64": torch.float64,
          "float32": torch.float32}[os.environ.get("GMM_MI_DTYPE", "float64")]
_REG = os.environ.get("GMM_MI_REG")
_EM_CHUNK = os.environ.get("GMM_MI_EM_CHUNK")


def _wrap(fn):
    def wrapped(*args, **kwargs):
        kwargs.setdefault("device", _DEVICE)
        kwargs.setdefault("dtype", _DTYPE)
        if _REG is not None:
            kwargs.setdefault("reg_covar", float(_REG))
        if _EM_CHUNK is not None:
            kwargs.setdefault("em_chunk", int(_EM_CHUNK))
        return fn(*args, **kwargs)
    return wrapped


gmm_mi_torch.estimate = _wrap(gmm_mi_torch.estimate)
gmm_mi_torch.estimate_categorical = _wrap(gmm_mi_torch.estimate_categorical)
