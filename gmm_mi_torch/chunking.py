"""Replicate and sample chunking: chunk sizes from the per-replicate memory
models, per-replicate weights read one sample chunk at a time, and the
CUDA out-of-memory halve-and-retry.
"""

import warnings

import torch

from .constants import (
    _AUTO_CHUNK_CPU_DEFAULT, _AUTO_CHUNK_MEM_FRACTION, _EM_KERNEL,
    _INIT_DRAW_BLOCK, _MSTEP_BLOCK)
from .precision import _check_chunk_arg


def _cuda_free_bytes(dev):
    """
    Return the bytes this process can still allocate on a CUDA device.

    The driver's free memory plus torch's cached-but-unallocated blocks (ours
    to reuse), capped by torch.cuda.set_per_process_memory_fraction when one is
    set (the allocator then refuses anything above fraction x total, so sizing
    against the device's free memory would open every stage with a
    guaranteed-OOM attempt).

    Parameters
    ----------
    dev : torch.device or str
        A CUDA device.

    Returns
    -------
    int
        The allocatable bytes, at least 0.
    """
    dev = torch.device(dev)
    free, total = torch.cuda.mem_get_info(dev)
    allocated = torch.cuda.memory_allocated(dev)
    free += torch.cuda.memory_reserved(dev) - allocated
    getter = getattr(torch.cuda, 'get_per_process_memory_fraction', None)
    if getter is not None:
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        frac = getter(idx)
        if frac < 1.0:
            free = min(free, int(frac * total) - allocated)
    return max(int(free), 0)


def _auto_chunk(n_items, bytes_per_item, device, default=_AUTO_CHUNK_CPU_DEFAULT,
                fixed_bytes=0):
    """
    Chunk size for `n_items` replicates costing `bytes_per_item` each.

    The budget is what remains after `fixed_bytes`, which the stage allocates
    once regardless of the chunk.

    `default` on CPU; on CUDA, as many items as fit in a fraction of the
    allocatable device memory (at least 1), capped at n_items.

    Parameters
    ----------
    n_items : int
        Number of replicates in the stage.
    bytes_per_item : int
        Peak memory per replicate, from the stage's memory model.
    device : torch.device or str
        Device the stage runs on.
    default : int, default=_AUTO_CHUNK_CPU_DEFAULT
        The chunk on CPU.
    fixed_bytes : int, default=0
        Memory the stage allocates once, whatever the chunk.

    Returns
    -------
    int
        Replicates per batch, between 1 and `n_items`.
    """
    dev = torch.device(device)
    if dev.type != 'cuda':
        return max(1, min(int(n_items), int(default)))
    # free memory counts torch's cache as ours; design notes: _auto_chunk
    free = _cuda_free_bytes(dev)
    bytes_per_item = max(int(bytes_per_item), 1)
    n = int((free * _AUTO_CHUNK_MEM_FRACTION - fixed_bytes) // bytes_per_item)
    return max(1, min(int(n_items), n))


def _resolve_chunk(value, n_items, bytes_per_item, device, fixed_bytes=0,
                   name='em_chunk'):
    """
    Turn an `em_chunk` or `mc_chunk` argument into an int.

    'auto' is sized from the stage's memory model; an int is checked and
    returned.

    Parameters
    ----------
    value : int or str
        The argument as given: 'auto' or a positive integer.
    n_items : int
        Number of replicates in the stage.
    bytes_per_item : int
        Peak memory per replicate, from the stage's memory model.
    device : torch.device or str
        Device the stage runs on.
    fixed_bytes : int, default=0
        Memory the stage allocates once, whatever the chunk.
    name : str, default='em_chunk'
        The argument's name, used in error messages.

    Returns
    -------
    int
        Replicates per batch.
    """
    value = _check_chunk_arg(value, name)
    if isinstance(value, str):
        return _auto_chunk(n_items, bytes_per_item, device,
                           fixed_bytes=fixed_bytes)
    return value


def _em_bytes_per_replicate(N, K, d, dtype, n_chunk=None):
    """
    Peak memory per EM replicate.

    From the measured model (design notes: _AUTO_CHUNK_MEM_FRACTION).

    Parameters
    ----------
    N : int
        Number of samples.
    K : int
        Number of components.
    d : int
        Number of variables.
    dtype : torch.dtype
        Working dtype.
    n_chunk : int, optional
        Sample chunk of the streamed path; None for the single pass.

    Returns
    -------
    int
        Bytes per replicate, for the kernels selected at import.
    """
    n_eff = N if n_chunk is None else min(N, int(n_chunk))
    if _EM_KERNEL == 'gemm':
        q = d + d * (d + 1) // 2
        units = max(K + 6, 2 * K + 3) + 3.0 * K * q / _MSTEP_BLOCK
    else:
        units = (4 * d + 4) * K + 4
    return int(1.03 * n_eff * dtype.itemsize * units)   # 3%: allocator rounding


def _em_fixed_bytes(N, d, dtype, n_chunk=None):
    """
    Memory an EM or val_ll stage allocates once, whatever the replicate chunk.

    The (N_eff, q) quadratic features of the GEMM path (the whole (N, q) in the
    single-pass path, one sample chunk's in the streamed); 0 for the legacy
    kernels.
    """
    if _EM_KERNEL != 'gemm':
        return 0
    n_eff = N if n_chunk is None else min(N, int(n_chunk))
    # the features at 2q units: the in-place assembly peaks at q + d(d+1)/2,
    # the rest is slack
    return 2 * n_eff * (d + d * (d + 1) // 2) * dtype.itemsize


def _init_sample_chunk(N, n_chunk=None):
    """
    Sample chunk of the streamed initialisation.

    n_chunk rounded down to a whole number of draw blocks (at least one block),
    capped at N; None (single pass) is the full N.
    """
    if n_chunk is None:
        return N
    return min(N, _INIT_DRAW_BLOCK * max(1, int(n_chunk) // _INIT_DRAW_BLOCK))


def _init_bytes_per_replicate(N, K, d, dtype, init_type, n_chunk=None):
    """
    Peak memory per replicate of the initialisation stage.

    The default and k-means inits run the broadcast M-step / assignment
    statistics over the full N without n_chunk, and over sample chunks of
    _init_sample_chunk(N, n_chunk) with it. minmax allocates nothing of note,
    so its figure is the EM model's (EM follows within the same budget).
    """
    if init_type == 'minmax':
        return _em_bytes_per_replicate(N, K, d, dtype, n_chunk)
    n_eff = _init_sample_chunk(N, n_chunk)
    # +12: the k-means pass holds the (c,n) int64 labels and the (c,n,K)
    # distance matrix beside the one-hot (docs/design_notes.md)
    return int(n_eff * dtype.itemsize * ((4 * d + 3) * K + 12))


def _mc_bytes_per_replicate(mc_samples, K, d, dtype):
    """
    Peak memory per Monte Carlo integration replicate, in bytes.

    From the model in the design notes (_AUTO_CHUNK_MEM_FRACTION).
    """
    return int(1.03 * mc_samples * dtype.itemsize
               * max((3 * d + 2) * K + d + 7, d * d + 4 * d + 6))


def _n_rows(s):
    """Replicate count: the product of the leading dims of `s`.

    `s` holds the sample weights, shape (..., n_samples).
    """
    return s.shape[:-1].numel()


def _rows(s, sl):
    """
    Select rows `sl` of the weights, along the flattened leading dims.

    A 2-D `s` is sliced (a view of the caller's tensor); an N-D `s` — typically
    an expand() view whose repeat structure (CV masks tiled over inits and
    candidate k) must never be materialised — is gathered into one (chunk, N)
    copy per replicate chunk.

    Parameters
    ----------
    s : torch.Tensor of shape (..., n_samples)
        Per-replicate weights.
    sl : slice
        Replicate range (step 1).

    Returns
    -------
    torch.Tensor of shape (chunk, n_samples)
        The rows, clipped to the replicate count.
    """
    if s.dim() == 2:
        return s[sl]
    idx = torch.arange(sl.start, min(sl.stop, _n_rows(s)), device=s.device)
    multi = []
    for size in reversed(s.shape[:-1]):
        multi.append(idx % size)
        idx = idx // size
    return s[tuple(reversed(multi))]


class _WeightChunks:
    """
    Per-replicate sample weights read one sample chunk at a time.

    Wraps the caller's weights `s` — 2-D (R, N), or (..., N) with the
    replicate axis over the leading dims, typically an expand() view of the
    CV masks — restricted to the replicate slice `sl`, so the streamed EM
    and scoring paths never hold an (R, N) copy: `rows(i, n, rows)` is the
    dense (|rows|, n) block of samples [i, i+n) for the given replicate
    rows of the slice (all of them for rows=None; a view for 2-D weights,
    one block-sized gather otherwise), and `totals()` the per-replicate
    weight sums accumulated over sample chunks at float64, cached. Row
    indices are absolute within the slice. The float64 total of per-chunk
    working-dtype sums is bitwise the full-row sum for the module's own
    weights (0/1 masks, resample counts, ones), and agrees with it up to
    the order of floating-point summation for real-valued weights passed
    directly to em_fit/val_ll.

    Parameters
    ----------
    s : torch.Tensor of shape (..., n_samples)
        Per-replicate weights.
    sl : slice
        The replicate range this accessor covers (clipped to the count).
    n_chunk : int, optional
        Sample chunk of `totals()`; None sums each row in one piece.
    """

    def __init__(self, s, sl, n_chunk=None):
        self.s = s
        self.start = sl.start
        self.R = min(sl.stop, _n_rows(s)) - sl.start
        self.N = s.shape[-1]
        self.n_chunk = self.N if n_chunk is None else int(n_chunk)
        if s.dim() == 2:
            self._idx = None
        else:
            idx = torch.arange(self.start, self.start + self.R,
                               device=s.device)
            multi = []
            for size in reversed(s.shape[:-1]):
                multi.append(idx % size)
                idx = idx // size
            self._idx = tuple(reversed(multi))
        self._tot = None

    @property
    def shape(self):
        return (self.R, self.N)

    def rows(self, i, n, rows=None):
        """
        Dense weights of samples [i, i + n) for some or all replicate rows.

        Parameters
        ----------
        i : int
            First sample.
        n : int
            Number of samples (clipped at the end of the sample axis).
        rows : torch.Tensor, optional
            Replicate rows within the slice; None for all of them.

        Returns
        -------
        torch.Tensor of shape (n_rows, n)
            A view for 2-D weights read without `rows`; otherwise one
            block-sized gather.
        """
        blk = self.s[..., i:i + n]
        if self._idx is None:
            blk = blk[self.start:self.start + self.R]
            return blk if rows is None else blk[rows]
        ix = self._idx if rows is None else tuple(t[rows] for t in self._idx)
        return blk[ix]

    def totals(self):
        """
        Per-replicate weight sums over all samples, at float64, cached.

        Returns
        -------
        torch.Tensor of shape (R,)
            The sums, accumulated over sample chunks of `n_chunk`.
        """
        if self._tot is None:
            tot = torch.zeros(self.R, dtype=torch.float64, device=self.s.device)
            for i in range(0, self.N, self.n_chunk):
                tot += self.rows(i, self.n_chunk).sum(1).double()
            self._tot = tot
        return self._tot


def _as_weight_chunks(s, n_chunk=None):
    """
    Accept a dense weight tensor where a `_WeightChunks` accessor is expected.

    For direct callers of the streamed steps: a dense (R, N) tensor becomes a
    view-only accessor over all of its rows (with `n_chunk` the sample chunk of
    its `totals()`); an accessor is returned as it is.
    """
    if isinstance(s, _WeightChunks):
        return s
    return _WeightChunks(s, slice(0, s.shape[0]), n_chunk)


def _legacy_sample_chunk(N, K, d, dtype, n_chunk=None):
    """
    Sample chunk for the centred legacy kernels on rerouted replicates.

    Their (4d+4)K-per-sample broadcast peak must fit three quarters of the GEMM
    budget the replicate chunk was sized with (the partition's chunk-sized
    gathers take the rest); at least 1. Shared by em_fit and val_ll so the two
    cannot drift apart.
    """
    return max(1, int(0.75 * _em_bytes_per_replicate(N, K, d, dtype, n_chunk))
               // (K * dtype.itemsize * (4 * d + 4)))


def _run_chunked(n, chunk, fn):
    """
    Apply a callback over replicate chunks, halving the chunk on CUDA OOM.

    fn(slice) runs over [0, n) in chunks of `chunk`; a CUDA out-of-memory error
    halves the chunk and retries. Results do not depend on the chunk on CPU
    (bitwise, with one float32 exception: see docs/estimate_notes.md,
    "Reproducibility"), so there the retry changes only peak memory; on CUDA
    they change only through the order of floating-point summation.

    Parameters
    ----------
    n : int
        Number of replicates.
    chunk : int
        Replicates per call to start with.
    fn : callable
        Called as fn(slice) for consecutive replicate ranges.

    Returns
    -------
    list
        The results of the calls, in replicate order.
    """
    outs, i = [], 0
    chunk = max(1, min(int(chunk), int(n)))
    while i < n:
        retry = False
        try:
            outs.append(fn(slice(i, i + chunk)))
        except torch.cuda.OutOfMemoryError:
            if chunk == 1:
                raise
            chunk = max(1, chunk // 2)
            retry = True
        if retry:
            # outside the handler: inside it the in-flight exception's
            # traceback still references the failed attempt's frame and
            # tensors, so an empty_cache there would release nothing
            torch.cuda.empty_cache()
            warnings.warn(f"CUDA out of memory; retrying with chunk={chunk}",
                          RuntimeWarning, stacklevel=3)
            continue
        i += chunk
    return outs
