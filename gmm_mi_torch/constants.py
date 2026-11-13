"""Module-level constants of gmm_mi_torch, and the EM kernel switch.

The environment variable GMM_MI_TORCH_EM_KERNEL ('gemm', the default, or
'legacy') is read here, at import; any other value raises ValueError. The
measurements behind the constants: docs/design_notes.md.
"""

import math
import os


# EM kernels, 'gemm' or 'legacy'; design notes: _EM_KERNEL
_EM_KERNEL = os.environ.get('GMM_MI_TORCH_EM_KERNEL', 'gemm')
if _EM_KERNEL not in ('gemm', 'legacy'):
    raise ValueError(f"GMM_MI_TORCH_EM_KERNEL must be 'gemm' or 'legacy'; "
                     f"found '{_EM_KERNEL}'")

LOG2PI = math.log(2.0 * math.pi)

# torch.multinomial's category cap: above it _bootstrap_counts histograms
# float64-uniform index draws instead. A module constant so tests can reach
# that path at small N by lowering it.
_MULTINOMIAL_MAX_CATEGORIES = 1 << 24

# per-replicate Monte-Carlo seed offsets; design notes: _MC_SEED_OFFSET
_MC_SEED_OFFSET = 50_000
_MC_CAT_STRIDE = 1 << 20

# bootstrap resample stream: seed + _BOOTSTRAP_SEED_OFFSET. In
# estimate_categorical, category ci runs its selection and its bootstrap
# on seed + _CATEGORY_SEED_STRIDE * ci (its Monte Carlo stream is set
# apart by _MC_CAT_STRIDE instead); design notes: "Random streams"
_BOOTSTRAP_SEED_OFFSET = 10_000
_CATEGORY_SEED_STRIDE = 997

# sample block of the default-init draw; design notes: _INIT_DRAW_BLOCK
_INIT_DRAW_BLOCK = 1 << 14

# gaussianise's random tie-breaking draws its permutations from
# seed + _TIE_SEED_OFFSET (one CPU generator, advanced per column so the
# per-column permutations are independent).
_TIE_SEED_OFFSET = 40_000

# auto-sized replicate chunks; design notes: _AUTO_CHUNK_MEM_FRACTION
_AUTO_CHUNK_MEM_FRACTION = 0.5      # of the allocatable device memory at call time
_AUTO_CHUNK_CPU_DEFAULT = 8


_MSTEP_BLOCK = 512
