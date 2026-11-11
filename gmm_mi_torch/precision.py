"""Full-precision float32 matmuls inside the kernels, and the validation of
the entry points' arguments.
"""

import functools
import operator

import torch

from .constants import _MC_CAT_STRIDE


# torch >= 2.9 per-backend precision settings; design notes: _FP32_KNOBS
try:
    _FP32_KNOBS = (torch.backends, torch.backends.cuda.matmul,
                   torch.backends.mkldnn.matmul)
    for _knob in _FP32_KNOBS:
        _knob.fp32_precision
except AttributeError:      # torch < 2.9: legacy API only
    _FP32_KNOBS = None


class _pinned_precision:
    """Context manager: float32 matmuls at full (ieee) precision inside the
    block, with any ambient torch.autocast region suspended (an enclosing
    mixed-precision context would otherwise run the cancellation-critical
    GEMMs at bf16/fp16, shifting both the estimate and the selected k).
    The caller's effective settings are restored on exit. Reentrant.

    Restore and inheritance (torch >= 2.9): the per-backend getters
    resolve an inherited ('none') knob to its parent's value, and torch
    exposes no raw getter, so a child knob is restored to 'none' whenever
    'none' reproduces the effective value recorded at entry, and to that
    value explicitly otherwise. A knob that inherited from the generic
    setting therefore keeps following it after the pin. The one case the
    rule cannot tell apart: a child set EXPLICITLY to the same value as
    its parent comes back inherited — the same precision now, and a later
    change of the parent then moves it too."""

    def __enter__(self):
        if _FP32_KNOBS is not None:
            self._prev = [knob.fp32_precision for knob in _FP32_KNOBS]
            for knob in _FP32_KNOBS:
                knob.fp32_precision = 'ieee'
        else:
            self._prev = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision('highest')
        # suspend autocast only for device types where it is active:
        # entering autocast('cuda', ...) on a CPU-only build warns
        self._autocast = []
        for dev_type in ('cuda', 'cpu'):
            try:
                active = torch.is_autocast_enabled(dev_type)
            except (TypeError, RuntimeError):
                active = False
            if active:
                cm = torch.autocast(dev_type, enabled=False)
                cm.__enter__()
                self._autocast.append(cm)
        return self

    def __exit__(self, *exc):
        for cm in reversed(self._autocast):
            cm.__exit__(*exc)
        if _FP32_KNOBS is not None:
            # the generic knob has no parent (recorded == raw); children
            # first try 'none' and keep it when that is the recorded
            # effective value, so inheritance survives the pin
            generic, children = _FP32_KNOBS[0], _FP32_KNOBS[1:]
            generic.fp32_precision = self._prev[0]
            for knob, prev in zip(children, self._prev[1:]):
                knob.fp32_precision = 'none'
                if knob.fp32_precision != prev:
                    knob.fp32_precision = prev
        else:
            torch.set_float32_matmul_precision(self._prev)
        return False


def _full_precision(fn):
    """
    Decorator: run `fn` with float32 matmuls pinned to full precision.

    Frameworks that set torch's matmul precision ('high'/'medium' via the
    legacy API, TF32/bf16 via the per-backend fp32_precision settings of torch
    >= 2.9) would silently degrade the GEMM-formulated EM. The pin uses the
    per-backend settings when torch has them (generic, cuda.matmul,
    mkldnn.matmul — the legacy getter raises once any of those has been set in
    the process) and the legacy get/set pair otherwise.

    Applied to estimate/estimate_categorical and, so that direct kernel
    calls are covered too, at the cancellation-critical kernel boundaries
    em_fit and val_ll (the pin is reentrant: nesting saves and restores
    the ambient values). These settings are PROCESS-GLOBAL, not
    thread-local: concurrent callers in other threads of the same process
    will also run these calls' matmuls at full precision, and the
    save/restore can race with theirs. Use processes, not threads, for
    parallel estimates (see examples/mi_grid_multigpu.py).

    Parameters
    ----------
    fn : callable
        The function to wrap.

    Returns
    -------
    callable
        `fn`, each call of which runs inside `_pinned_precision()`.
    """
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with _pinned_precision():
            return fn(*args, **kwargs)
    return wrapped


def _entry_precision(fn):
    """
    Decorator for the public estimators: full precision and no autograd.

    _full_precision plus torch.no_grad(). The public estimators return plain
    floats, so an input tensor carrying requires_grad (model latents taken
    without .detach()) would otherwise build and retain the EM autograd graph
    of every iteration — several times the peak memory — and crash FittedGMM's
    .numpy() calls under return_fit=True. em_fit and val_ll keep plain
    _full_precision: differentiating a log-likelihood with respect to the
    parameters is legitimate.

    Parameters
    ----------
    fn : callable
        The function to wrap.

    Returns
    -------
    callable
        `fn`, each call of which runs inside `_pinned_precision()` and
        `torch.no_grad()`.
    """
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with _pinned_precision(), torch.no_grad():
            return fn(*args, **kwargs)
    return wrapped


def _check_chunk_arg(value, name):
    """
    Validate an `em_chunk` or `mc_chunk` argument.

    'auto' is returned as is, anything else coerced to an int >= 1. A float
    (2.7) fails under the argument's name like every other count.

    Parameters
    ----------
    value : int or str
        The argument as given: 'auto' or a positive integer.
    name : str
        The argument's name, used in the error message.

    Returns
    -------
    int or str
        'auto', or `value` as a plain int >= 1.
    """
    if isinstance(value, str):
        if value != 'auto':
            raise ValueError(f"{name} must be a positive int or 'auto'; "
                             f"found '{value}'")
        return value
    return _index_arg(value, name, 1)


def _index_arg(value, name, minimum=None):
    """
    Coerce an integer argument at entry.

    operator.index accepts numpy integers (the natural output of numpy RNGs in
    seed sweeps) and rejects floats, so a malformed count fails here under the
    argument's own name instead of deep inside EM with an opaque torch error
    after the compute has already been spent.

    Parameters
    ----------
    value : int
        The argument as given (a Python or numpy integer).
    name : str
        The argument's name, used in the error messages.
    minimum : int, optional
        Smallest accepted value; None accepts any integer.

    Returns
    -------
    int
        `value` as a plain Python int.
    """
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer; found {value!r} of "
                        f"type {type(value).__name__}") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; found {value}")
    return value


def _as_array(X):
    """
    Unwrap array-protocol containers that torch.as_tensor cannot inspect.

    A pandas DataFrame exposes __array__ but torch's sequence path fails on
    it with "could not determine the shape of object type 'DataFrame'".
    The container converts itself, so no numpy import is needed, and a
    plain ndarray returns itself (no copy). Read-only arrays (pandas
    copy-on-write hands those out) are copied so torch.as_tensor does not
    warn about non-writable memory.

    Parameters
    ----------
    X : array-like or torch.Tensor
        The data as given by the caller.

    Returns
    -------
    array-like or torch.Tensor
        The container's array (a writable copy if it was read-only), or `X`
        itself when it is a tensor or has no __array__.
    """
    if not isinstance(X, torch.Tensor) and hasattr(X, '__array__'):
        X = X.__array__()
        if getattr(getattr(X, 'flags', None), 'writeable', True) is False:
            X = X.copy()
    return X


def _validate_entry_args(seed, n_bootstrap, mc_samples, max_components,
                         n_folds, n_inits, max_iter,
                         fixed_components_number, reg_covar,
                         threshold_components, threshold_fit=1e-5,
                         patience=1, n_chunk=None, mc_chunk='auto'):
    """
    Validate the arguments shared by `estimate` and `estimate_categorical`.

    Shared, so the two cannot drift apart. Returns the integer arguments
    coerced to plain Python ints (torch.Generator.manual_seed and torch factory
    sizes reject numpy integers and floats). mc_chunk is only checked here: its
    'auto' sizing needs the selected k, so the caller resolves it later.

    Parameters
    ----------
    seed, n_bootstrap, mc_samples, max_components : int
        As in `estimate`.
    n_folds, n_inits, max_iter : int
        As in `estimate`.
    fixed_components_number : int or None
        As in `estimate`.
    reg_covar, threshold_components : float
        As in `estimate`.
    threshold_fit : float, default=1e-5
        As in `estimate`.
    patience : int, default=1
        As in `estimate`.
    n_chunk : int, optional
        As in `estimate`.
    mc_chunk : int or str, default='auto'
        As in `estimate`; checked, not resolved.

    Returns
    -------
    tuple
        (seed, n_bootstrap, mc_samples, max_components, n_folds, n_inits,
        max_iter, fixed_components_number, patience, n_chunk) as plain
        Python ints (fixed_components_number and n_chunk may be None).
    """
    seed = _index_arg(seed, 'seed')
    if abs(seed) > 1 << 62:
        raise ValueError(f"|seed| must be <= 2**62 so the internal stream "
                         f"offsets stay within torch.Generator.manual_seed's "
                         f"range; found {seed}")
    n_bootstrap = _index_arg(n_bootstrap, 'n_bootstrap', 0)
    if n_bootstrap > _MC_CAT_STRIDE:
        raise ValueError(f"n_bootstrap must be <= {_MC_CAT_STRIDE} (the "
                         f"per-replicate MC stream stride); found "
                         f"{n_bootstrap}")
    mc_samples = _index_arg(mc_samples, 'mc_samples', 1)
    max_components = _index_arg(max_components, 'max_components', 1)
    n_folds = _index_arg(n_folds, 'n_folds')   # >= 2 checked by _validate_folds
    n_inits = _index_arg(n_inits, 'n_inits', 1)
    max_iter = _index_arg(max_iter, 'max_iter', 1)
    if fixed_components_number is not None:
        fixed_components_number = _index_arg(
            fixed_components_number, 'fixed_components_number', 1)
    if not reg_covar > 0:
        raise ValueError(f"reg_covar must be > 0 (the covariance floor; 0 "
                         f"lets collapsed components through to garbage "
                         f"fits or factorisation failures); "
                         f"found {reg_covar}")
    if not threshold_components >= 0:
        raise ValueError(f"threshold_components must be >= 0; found "
                         f"{threshold_components}")
    # threshold_fit <= 0 (or NaN) can never satisfy the EM stop test, so
    # every replicate of every stage would run max_iter; a non-number
    # would die in the stop test only after the init stage had run
    try:
        ok = threshold_fit > 0
    except TypeError:
        raise TypeError(f"threshold_fit must be a positive number; found "
                        f"{threshold_fit!r} of type "
                        f"{type(threshold_fit).__name__}") from None
    if not ok:
        raise ValueError(f"threshold_fit must be > 0 (the EM convergence "
                         f"threshold); found {threshold_fit}")
    patience = _index_arg(patience, 'patience', 1)
    if n_chunk is not None:
        n_chunk = _index_arg(n_chunk, 'n_chunk', 1)
    _check_chunk_arg(mc_chunk, 'mc_chunk')
    return (seed, n_bootstrap, mc_samples, max_components, n_folds, n_inits,
            max_iter, fixed_components_number, patience, n_chunk)
