"""MI over all variable pairs, one worker process per GPU.

Run as a script (this cannot be pasted into a notebook or an interactive
Python session: the workers are started with the 'spawn' method, which
re-imports this file, and an interactive session has no file to re-import —
the symptom is "AttributeError: Can't get attribute ... on <module
'__main__'>" followed by a BrokenProcessPool):

    python examples/mi_grid_multigpu.py

Replace load_data() with your own data. On a single-GPU machine this runs
with one worker, which is no faster than a plain loop over estimate().
"""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch


def load_data():
    # stand-in for your own (n_samples, n_variables) array
    rng = np.random.default_rng(0)
    return rng.multivariate_normal(
        [0, 0, 0], [[1, 0.6, 0.2], [0.6, 1, 0.4], [0.2, 0.4, 1]], 10_000)


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
        raise SystemExit("no CUDA device is visible; use a plain loop with "
                         "estimate(..., device='cpu') instead")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = mask.split(",") if mask else [str(i) for i in range(n_gpus)]
    X = load_data()
    pairs = [(i, j) for i in range(X.shape[1])
             for j in range(i + 1, X.shape[1])]
    tasks = [((i, j), X[:, [i, j]]) for i, j in pairs]
    ctx = mp.get_context("spawn")   # fork is unsafe once CUDA is involved
    counter = ctx.Value("i", 0)
    with ProcessPoolExecutor(max_workers=n_gpus, mp_context=ctx,
                             initializer=init_worker,
                             initargs=(counter, visible)) as pool:
        for i, j, mi, sigma in pool.map(one_pair, tasks):
            print(f"MI(x{i}, x{j}) = {mi:.4f} +- {sigma:.4f}")
