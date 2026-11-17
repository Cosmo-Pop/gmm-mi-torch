# Design notes

The mechanisms and measurements behind the `gmm_mi_torch` package. The
docstrings state what each function promises; this file records why the code
is shaped as it is and the measurements behind its constants;
[CHANGELOG.md](../CHANGELOG.md) records what changed in which release.
Measurements are on the NVIDIA GB10 (128 GB unified memory) unless stated.

## Architecture

- The code is split along its stages (since 0.10.0; one file,
  `estimator.py`, before): `constants` (module constants and the
  `GMM_MI_TORCH_EM_KERNEL` switch), `precision` (full-precision float32
  matmuls, entry validation), `chunking` (memory models, chunk sizes,
  per-replicate weights, the out-of-memory retry), `gaussian` (Cholesky
  with repair, densities, sampling, marginals), `em` (the GEMM and legacy
  EM kernels, the initialisations, `em_fit`, `val_ll`), `selection` (CV
  masks, the two protocols) and `estimator` (integrands, preprocessing,
  bootstrap, `FittedGMM`, `estimate`, `estimate_categorical`). Each module
  imports from the ones before it; `estimator` re-exports the public names
  it defined before the split.
- No sklearn. EM is implemented as *weighted* EM over a leading replicate
  dimension R. Per-sample weights unify everything that GMM-MI does with
  loops and index resampling:
  - CV folds → 0/1 masks over the full dataset (train mask; complement
    scores validation LL), so unequal fold sizes are free;
  - bootstrap → multinomial(N, 1/N) counts as weights (identical in
    distribution to index resampling);
  - inits → just more replicates.

  One EM loop therefore fits (n_inits × n_folds), or B bootstrap
  realisations, or (K_max × n_inits × n_folds), as a single batched tensor
  computation.
- Marginalisation is analytic index masking of the joint GMM, as in GMM-MI,
  so MI / conditional MI / total correlation all come from one joint fit and
  the marginals are exactly consistent with the joint.
- Two component-selection protocols:
  - `'sequential'`: GMM-MI's protocol (increment k, validation-LL metric,
    improvement `threshold_components` + patience early stop);
  - `'batched'`: fit k = 1..K_max simultaneously (dead-component padding to
    a common tensor shape), argmax validation LL.

## GEMM kernels and the kernel reroute

The default EM kernels evaluate log w_k N(x | mu_k, Sigma_k) as one matrix
product of per-sample quadratic features F(x) = [x, x_i x_j (i ≤ j)] with
per-component coefficients, and the M-step moments as the transposed product
resp^T F (Machlica, Vaněk & Zajíc 2011). The expanded quadratic form cancels
almost completely for points inside a component, so the coefficients are
assembled and the moments accumulated at float64, and the data are
standardised at entry (Chan, Golub & LeVeque 1983). The precision
argument in full: "GEMM-formulated EM kernels" under "Notes on
individual constants and functions" below. GEMM is the general matrix
multiply; the legacy kernels are the second implementation of the EM step,
selected with `GMM_MI_TORCH_EM_KERNEL=legacy`, which subtracts each
component's mean from the samples before evaluating the quadratic form.

### Kernel reroute

Standardisation bounds |x| but not the rounding residue relative to a
collapsed component's variance, so replicates carrying a component collapsed
to the `reg_covar` floor away from the origin are detected and finished on
the centred legacy kernels. The detector uses the min diagonal as the
precision proxy and is deliberately blind to rho → 1 components with O(1)
diagonals. The measurements behind that choice: the detector understates
such a component's precision P = 1/lambda_min (lambda_min the smallest
eigenvalue of its covariance) by diag/lambda_min, measured ×500 at
rho = 0.999; there
the GEMM float32 kernels stay within 1 sigma of float64 (residual
+0.016..0.025 nats at |mu|^2 ~ 8-20 with lambda_min = reg_covar) while legacy
float32 is measurably WORSE (repeated covariance repairs, the failure seen
in beta testing at v0.2.x),
so such replicates must not be rerouted. Tests pin both sides.

### M-step moment GEMM

At float32 the moments are accumulated as block-wise float32 GEMMs (blocks of
512 samples) reduced in float64, batched over blocks only with (k, r) folded
into the GEMM's row dimension, so F is never broadcast. The previous form
batched torch.matmul over (R, nb) against a (1, nb, b, q) F, which expanded
and copied F R times (q/K responsibility buffers every M-step, 23x at d=8,
K=2) and copied wr too whenever N % block != 0. The tail block stays a 3-D
broadcast matmul that copies nothing: a folded-row tail would add a copy of
its own and change the tail-only regime's summation order.

## Memory models

`em_chunk='auto'` and `mc_chunk='auto'` size each stage's replicate batch on
CUDA from per-replicate peak-memory models, in units of N_eff × itemsize per
replicate (N_eff = min(N, n_chunk)):

| stage | model (units of N_eff × itemsize, with one to two units of slack over the measured peaks) | what dominates |
|---|---|---|
| GEMM EM step | max(K+4, 2K) + 3Kq/block | E-step: one (R,N,K) buffer plus four (R,N) temporaries; M-step: the buffer plus its one blocked copy and the block outputs; q = d + d(d+1)/2 |
| legacy EM step | (4d+4)K + ~2 | broadcast (R,N,K,d) operands; the rerouted replicates stream their samples so they fit the GEMM budget |
| init stage | (4d+3)K + 12 | default and k-means inits over N, or over the n_chunk-rounded sample chunk when streamed; ~0 for minmax |
| MC stage | max((3d+2)K + d + 7, d^2 + 4d + 6) per mc_sample | integrand with its analytic marginals; gmm_sample's (S,d,d) Cholesky gather dominates at small K |

The functions that size the batches (`_em_bytes_per_replicate` and
siblings) use the init and MC rows exactly
and larger EM terms — GEMM max(K+6, 2K+3) + 3Kq/512, legacy (4d+4)K + 4;
the EM and MC functions multiply by 1.03 for allocator rounding (the init
function does not; that is, for the default and k-means inits — 'minmax'
uses the EM model); the GEMM path adds a per-stage fixed term of 2q units
for the quadratic features; and `estimate_categorical`'s MC stage adds 4
units per mc_sample for its running logaddexp over categories.

- **Provenance.** Re-derived over K = 1..20 and d = 1..8
  (`validation/bench_mem_models.py`: 207 measured rows per dtype, run at fp32
  and with `--fp64` as of 2026-10-23, every stage incl. val_ll, the streamed
  paths with expand-view masks, the init stage and the d=1 MC stage).
- **The one exception.** Every row is at or below its model except the
  streamed CV-mask em_fit rows at N=4e5, d=2, where a fixed ~10-19 MiB cuBLAS
  workspace shows against 5-45 MiB models: 1.06 at K=3 fp32, 1.42 at K=20
  with n_chunk=4096 at both dtypes — an absolute term, irrelevant at real
  budgets, and not added to the model.
- **Init stage +12.** The k-means pass holds the (c,n) int64 labels and the
  (c,n,K) distance matrix beside the one-hot — measured 1.7x an 11 MiB model
  at K=1, d=1 with the old +4.
- **Responsibilities released before the next E-step.** Left bound, the
  consumed responsibilities would stay allocated through the next E-step and
  double the (R,N,K) peak (measured 2K+4 -> K+4 units).
- **Init draw block.** The default initialisation draws each replicate's
  responsibilities in fixed blocks of 2^14 samples; 2^14 measured on the
  GB10: 1.3 ms per replicate at N=1e6 (one-shot 0.6 ms), ~0.1 s per replicate
  and pass at N=1e8.
- **Bootstrap counts.** The multinomial path's draw is the validated resample
  stream, so it stays ONE call on the whole (B, N) index matrix — but its
  probability input is a single (1, N) row expanded to (B, N) (bitwise the
  same draw as the dense matrix on CPU and CUDA, measured) and the scatter
  source an expanded ones row, so the only full-size transient is
  torch.multinomial's own (B, N) int64 index matrix plus its internal prefix
  sums (CUDA, float64 row: ~4x the float64 weights, i.e. ~8x float32
  weights; outside the OOM guard by design).
- **Bootstrap counts above 2^24.** The large-N path (N > 2^24,
  where the weights alone are GiB) draws and histograms one replicate at a
  time from per-replicate generators: its transient is one (N,) float64
  uniform vector and the (N,) int64 index vector it becomes, 16 B/sample
  measured, written as one expression so both are released before the next
  replicate's draw (binding them across iterations measures 24 B/sample).

## Bootstrap resample

`_bootstrap_counts` draws the uniform multinomial resample counts. Up to
torch.multinomial's 2^24-category cap they come from one torch.multinomial
call; above it, from per-replicate float64-uniform index draws floor(u N).

- **Probability row at float64.** The probability row is built at float64
  whatever the working dtype. torch's CPU multinomial accumulates the row's
  CDF sequentially in the row's dtype, and at float32 the step 1/N rounds
  differently in each binade of the running sum: a fixed,
  replicate-independent tilt of the sampling weights by index region (16
  sigma at N=1e6, up to 33% max/min at N=1e7 — a bias on index-ordered data
  that the bootstrap errorbar cannot see). With the float64 row the float32
  and float64 resamples are uniform and bitwise identical to each other, on
  CPU and CUDA.
- **CUDA multinomial (N ≤ 2^24): float32 resolution.** On CUDA,
  torch.multinomial's kernel draws float32 uniforms even for a float64 row,
  so each sample's inclusion weight carries a fixed, replicate-independent
  factor set by how many 2^-24 grid points its 1/N interval holds — no
  regional tilt (index-band means within 4e-4 of 1), but a per-sample weight
  scatter sd(w) of 0.019 at N=1e6, 0.043 at 2e6, 0.073 at 4e6, 0.113 at 8e6,
  peaking ~0.24 near N ~ 1.1e7 and vanishing at exactly 2^24 (closed form:
  per float32 binade k with m_k = 2^(24+k)/N grid points per interval,
  variance 2^-(k+1) frac(m_k)(1 - frac(m_k))/m_k^2).
- **What the scatter does.** It is a fixed shift of the bootstrap mean of
  order sd(w) × one standard error; the CPU path has none (excess variance
  -1e-5 at N=1e6). This stream is kept because it is
  the validated resample stream (CHANGELOG 0.7.0).
- **Large-N draw (N > 2^24): uniformity and the clamp.** The index draw
  floor(u N) from float64 uniforms (which replaced a 32-bit modulo randint in
  0.7.0; see CHANGELOG) is uniform to ≤ N/2^53 on CPU, whose uniforms lie on
  the 2^-53 grid, and to ≤ N/2^52 on CUDA, where the kernel's half-step
  values above 0.5 round to even onto the 2^-52 grid (measured at torch
  2.11): 2.2e-8 at N=1e8, with no regional structure.
- **The clamp.** The clamp to N - 1 guards only a kernel returning u = 1.0 (torch reflects it to 0.0 today),
  which would otherwise raise in scatter_add_: for every float64 u < 1 and
  N ≤ 2^53, floor(fl(u N)) ≤ N - 1, since the deficit N 2^-53 is at least
  half an ulp of N, with equality only at powers of two, where the product
  is exact.
- **CPU per-replicate seed space.** The per-replicate seeds — of the large-N
  draw and of the default init — are drawn from [0, 2^62), but CPU
  torch.Generator.manual_seed keeps 32 bits (CUDA keeps 64), so the CPU
  per-replicate seed space is 2^32: two replicates of one call share a stream
  with probability R(R-1)/2^33 (1.7e-6 at the default 120 replicates).

## Random streams

Every stream derives from the caller's `seed` by a fixed offset, so the
streams are decoupled from each other (compiled from the module constants):

| stream | generator seed |
|---|---|
| initialisation, sequential protocol and fixed k (candidate k = K) | seed + K |
| initialisation, batched protocol | seed |
| default init, replicate r | per-replicate seed drawn from the init generator |
| bootstrap resample | seed + 10_000 (`_BOOTSTRAP_SEED_OFFSET`) |
| bootstrap above 2^24, replicate r | per-replicate seed drawn from the bootstrap generator |
| CV fold assignment (CPU generator) | seed + 30_000 (`_CV_MASK_SEED_OFFSET`) |
| gaussianise tie-breaking (one CPU generator, advanced per column) | seed + 40_000 (`_TIE_SEED_OFFSET`) |
| Monte-Carlo integration, replicate r | seed + 50_000 + r (`_MC_SEED_OFFSET`) |
| `estimate_categorical`, category ci: selection and bootstrap | as above with seed + 997 ci (`_CATEGORY_SEED_STRIDE`) |
| `estimate_categorical`, category ci: MC, replicate r | seed + 50_000 + ci 2^20 + r (`_MC_CAT_STRIDE` = 2^20) |

n_bootstrap is capped at 2^20 and |seed| at 2^62 at entry so the offsets stay
inside torch.Generator.manual_seed's range and category streams cannot
collide. One known, low-risk stream coincidence remains, present since
before 0.4.0: category ci+10's
init stream at K=30 equals category ci's bootstrap stream (997·10 + 30 =
10_000); it needs ≥ 11 categories AND max_components ≥ 30 (default 10) and
the streams drive different data, so it is documented rather than re-seeded
under the validated init/bootstrap offsets.

## Reproducibility on CUDA

Fixed-seed results are bitwise reproducible on CPU at float64 whatever the
chunking; at float32 on CPU also bitwise except em_chunk=1 with a
one-component model, where the M-step's single-row tail product takes a
different BLAS path (3e-8 in a measured MI; details below). On CUDA the
replicate chunk `em_chunk` changes the matmul reduction order:

- At float32 the blocked M-step folds the batch's replicates into one GEMM,
  whose summation order follows the batch size, so the chunk — including an
  out-of-memory retry, which pinning em_chunk does not prevent — moves
  results at reduction order, and on knife-edge data that can move the
  selected k (measured: k=5 vs 6, MI 0.4741 vs 0.4745 against an errorbar of
  0.008).
- At float64 the chunk also moves results, at the reduction-order level
  (2026-10-24, v0.7.0, torch 2.11, two probes): an em_fit of 24 bootstrap
  replicates (N=40,000, d=2, K=6, 50 iterations) gave fitted means
  differing by 8.3e-14 between em_chunk=24 and 7 (8.1e-14 for 3), against
  6.8e-3 at float32; a second em_fit (N=20,000, 12 replicates,
  K=4, 40 iterations) gave 1.2e-14 for em_chunk 5 vs 12 and 1.0e-12 for 1
  vs 12 (float32: 2.6e-6 and 3.1e-3). CPU was bitwise for every chunk at
  both dtypes in these two probes (K=6 and K=4).
- The same holds for the other chunked stages on CUDA:
  val_ll with em_chunk 1 vs 12 differed by 4.4e-16 at float64 and 2.4e-7 at
  float32; the default initialisation's statistics by 3.6e-15 and 9.0e-6
  (its draw is chunk-invariant by construction); an estimate with em_chunk
  pinned at 4 and mc_chunk 10, 3 or 1 gave standard deviations differing by
  5e-18 to 2e-17 at float64 and means by 3.0e-8 at float32 — the MC
  samples are identical, the integrand's reduction order is not. CPU was
  bitwise throughout in these probes.
- The CPU float32 exception (measured 2026-10-24, at v0.7.1): with a one-component model and em_chunk=1,
  `_moment_gemm`'s tail term (the N % 512 samples after the last full
  block) is a (1, 1, tail) @ (1, tail, q) product, which CPU float32 BLAS
  evaluates on a different path (a different summation order) from the
  same row inside a multi-row batch; the blocked part is bitwise. An
  estimate at N=3000 (bivariate Gaussian, max_components=3, 8 bootstrap
  replicates, mc_samples=5000, seed=3) gave MI means differing by 3.0e-8
  between em_chunk=1 and 8 (the same with OMP_NUM_THREADS=1); em_chunk 2,
  4 and 16 were bitwise equal to 8; only em_fit calls with K=1 differ; a
  direct em_fit at K=1, R=8 differed by 3.7e-9 in the fitted means, K=2 and
  K=3 bitwise; N=3072 (no tail), float64, the legacy kernels, val_ll and
  the initialisation were bitwise. An independent re-run (8 replicates
  converging at different iterations, N=3000) reproduced 3.7e-9 at
  em_chunk=1 and bitwise results at em_chunk 2 and 4, and at N=3072.

## Precision pin and threads

`estimate`, `estimate_categorical`, `em_fit` and `val_ll` pin float32 matmuls
to full precision (and suspend any ambient torch.autocast region) for the
duration of the call. These settings are PROCESS-GLOBAL, not thread-local:
concurrent callers in other threads of the same process will also run these
calls' matmuls at full precision, and the save/restore can race with theirs.
Use processes, not threads, for parallel estimates (see
`examples/mi_grid_multigpu.py`).

## torch 2.13 and one-dimensional fits on CUDA

torch 2.13.0 overrides the CUDA batched matrix product for operands of
shapes (B, n, 1) and (B, 1, m) with a kernel written in Triton
(`torch/_native/ops/bmm_outer_product`), which is compiled at first use.
The compilation builds a small C module that includes `Python.h`; where the
header is missing, torch raises `subprocess.CalledProcessError` and does not
fall back to its ordinary product. A bare `torch.bmm` of those shapes fails
in the same way. torch keeps the compiled kernel in a cache
(`~/.triton/cache` by default), so the failure appears only where the
kernel has not been compiled before.

A 1×1 product has those shapes, so a fit with d = 1 reaches the kernel
through three products: `Y^T Y` and `P mu` in `_gauss_coef`, and
`chol eps` in `gmm_sample`. With the default kernels, fits with d = 2 and
d = 3 (MI and KL; cMI and TC) did not reach it; the legacy kernels were not
tested.

Measured on the GB10 with torch 2.13.0+cu130, GPU float32 suite of 133
tests: 118 passed, 14 failed and 1 skipped without the header, every
failure in a test that calls `estimate_categorical`; 132 passed and 1
skipped with the header on the compiler's search path; and 132 passed and 1
skipped with `TORCH_DISABLE_NATIVE_JIT=1`, torch's switch that leaves the
override unregistered. Under torch 2.11.0+cu128 the same fits run without
either.

## Notes on individual constants and functions

The longer explanations behind individual constants and functions, in
the order the code reaches them. Each subsection is named after the
function or constant it explains; the code carries one line saying what
the next lines do and naming the subsection. "The kernel section"
means "GEMM-formulated EM kernels" below.

### `_EM_KERNEL`

EM kernel implementation. 'gemm' evaluates the Gaussian log-densities
and the M-step moments as matrix products against per-sample quadratic
features (see the kernel section below); 'legacy' selects the previous
broadcast kernels. The two agree at reduction-order precision wherever
the GEMM cancellation residue is below the reg_covar floor (reg_covar >=
~1e-13 at float64; at float32, standardised data with the reroute
guarding collapsed components — see the kernel section). A typo must
fail here, not silently select the legacy kernels.

### `_FP32_KNOBS`

torch >= 2.9 replaces get/set_float32_matmul_precision with per-backend
fp32_precision settings; once user code has set any of those, the legacy
getter RAISES ("mix of the legacy and new APIs"). The per-backend settings
are also what make an exact restore possible: the legacy setter rewrites
every backend at once, so restoring through it would hand a caller who had
only enabled CUDA TF32 a changed mkldnn (CPU) setting.

### `_MC_SEED_OFFSET`

Per-replicate Monte-Carlo streams: bootstrap replicate r draws its MC
samples from a generator seeded seed + _MC_SEED_OFFSET + r (estimate), or
\+ ci * _MC_CAT_STRIDE + r per category ci (estimate_categorical), so the
draws are independent of mc_chunk and of the OOM halve-and-retry by
construction. n_bootstrap is capped at _MC_CAT_STRIDE and |seed| at 2**62
at entry so the offsets stay inside torch.Generator.manual_seed's range
and category streams cannot collide.

### `_INIT_DRAW_BLOCK`

The default initialisation draws each replicate's responsibilities in
fixed sample blocks of this many rows, sequentially from the replicate's
own generator, on the single-pass and the streamed (n_chunk) path alike:
the draw is then invariant to em_chunk AND n_chunk by construction on
every device (CUDA's Philox streams are not sequential across calls, so
any other chunking of the draw would depend on the chunk size). Timings
of the 2^14 block: "Memory models" above.

### `_AUTO_CHUNK_MEM_FRACTION`

Auto-sizing of the replicate chunks (em_chunk / mc_chunk = 'auto'). The
per-replicate peak-memory models are measured on the GB10 over
K = 1..20 and d = 1..8 by validation/bench_mem_models.py; the models, the
bench and its one known exception are under "Memory models" above. On
CUDA a stage's batch is whatever fits the budget fraction of the memory
this process can still allocate; there is no minimum batch size. On CPU
the batch is 8 replicates.

### `_auto_chunk`

The driver counts torch's cached but unallocated blocks as used, although
this process can reuse them, so the allocatable memory is taken as the
driver's free figure plus torch's cache. Without the cache, every stage
after the first (and every later call in the process) would be sized
against free - cache, which is about half the intended batch when memory
is the binding constraint, and would never recover.

### GEMM-formulated EM kernels

log w_k N(x|mu_k, Sigma_k) is a quadratic form in x, so against per-sample
features F(x) = [x, x_i x_j (i<=j)] it is one matrix product with
per-component coefficients, and the M-step moments are the transposed
product resp^T F (Machlica, Vanek & Zajic 2011). The only tensor of shape
(n_replicates, n_samples, n_components) is the responsibilities, plus
one transient float32 copy of it inside the blocked M-step (see
_moment_gemm).
The expanded quadratic form cancels almost completely for points inside a
component, so the coefficients are assembled at float64 and the moments
accumulated at float64 (see _gauss_coef and _moment_gemm); the data are
standardised in estimate(), which bounds the magnitudes entering the
cancellation (Chan, Golub & LeVeque 1983).

Standardisation bounds |x| but NOT the residue relative to a collapsed
component's variance: the uncentred form leaves a rounding residue
~ u * P_k * |x|^2 in the log-density and ~ u * |mu_k|^2 per entry of the
M2 - m1 m1^T covariance (u = unit roundoff), so a component at the
reg_covar floor away from the origin is beyond these kernels' precision
at float32 (and below reg_covar ~1e-13 at float64). Replicates in that
regime are detected and finished on the centred legacy kernels (see
`_REROUTE_TOL_NATS` below, which also states which rho->1 components the
detector deliberately does not flag). Direct
em_fit/val_ll calls on unstandardised
float32 data fall in the detected regime too: they run correctly, via
the slower legacy path.

### `_init_params`

The (R,N,K) responsibility draw is the largest allocation of the
selection stage (R = max_components x n_inits x n_folds), so it
is drawn per replicate chunk inside the OOM-guarded callback, in
place (mask, normalise) so no second full-size transient exists.
Each replicate draws from its own generator seeded from one
value taken off the caller's stream: the draw is then invariant
to em_chunk and to the halve-and-retry by construction on every
device (CUDA's Philox streams are NOT sequential across calls,
so chunked draws from one generator would depend on the chunking).
CPU manual_seed keeps 32 of these 62 bits (CUDA all 64): the CPU
per-replicate seed space is 2^32, collision R(R-1)/2^33.

### `_REROUTE_TOL_NATS`

Kernel reroute: the GEMM kernels evaluate the Gaussian quadratic form and
the M-step covariance from uncentred origin moments, whose rounding
residue scales as u * P_k * |x|^2 (u = unit roundoff). For a component
collapsed to the reg_covar floor away from the origin this swamps both
the log-density (O(1)-nat errors at float32) and the covariance
(indefinite every iteration), silently biasing MI on spike-bearing,
zero-inflated or duplicated-point data — and at float64 it defeats
reg_covar <= ~1e-13. The centred legacy kernels are immune, so replicates
whose parameters make the residue material are finished on them. The
detector uses the min diagonal as the precision proxy: it trips on every
executed failure case and never on continuous data, but is DELIBERATELY
blind to rho->1 components with O(1) diagonals: there the GEMM float32
kernels stay within 1 sigma of float64 while legacy float32 is
measurably WORSE, so such replicates must NOT be rerouted; tests pin
both sides (measurements: "Kernel reroute" above).

### `_reroute_trips`

A component whose factorisation needed extra regularisation is further
evidence of the cancellation, but regularisation alone is also needed for
rho->1 components, where the legacy float32 kernels are the worse choice.
A repaired component therefore only lowers the threshold tenfold, and
only for itself: a repaired rho->1 component does not cause its replicate
to be rerouted on the strength of another component's statistic.

### `_em_fit_core`, compaction

Only the replicates that have not yet converged stay in the working
batch. On the streamed path their weights are addressed through
`active_idx`; on the single-pass path the dense `weights_active` is
reduced alongside.

### `_em_fit_core`, reroute mask

The per-replicate reroute mask covers the active set, and a replicate
once rerouted stays rerouted. The legacy step streams its samples in
chunks of `legacy_sample_chunk`, so that its peak of (4d+4)K per sample
stays inside the GEMM budget with which the replicate chunk was sized.

### `_val_ll_core`

Rerouted rows are scored by the centred legacy kernels in the same
model-derived sample chunk in which em_fit streams them; one broadcast
pass over N would exceed, several times over, the budget with which the
replicate chunk was sized.

### `_SMALL_SAMPLE_WARN`

Below this many samples (per dataset, or per category) the estimator
warns: the GMM MI estimate carries a positive finite-sample bias that the
bootstrap errorbar does not include, and the validation campaign started
at N ~ 1e3.

### `_GAUSSIANISE_TIE_WARN_FRACTION`

Tie fraction (excess samples in tied blocks of size >= 3, per column)
above which gaussianise warns. Blocks of two are ignored: genuinely
continuous data acquires birthday-pair collisions once quantised (a
float32-native standard-normal column crosses 1% of paired samples around
N ~ 2e6), and a two-sample block is harmless jitter, while real
quantisation or zero inflation shows up as large blocks.

### `_rank_gaussianise`

The ranks are positional ranks after a seeded random row permutation.
With a stable sort, the order within each tied block is exactly the
permutation's, that is uniformly random, while untied values keep their
exact ranks.

### `_select_and_fit`

The batched protocol pads parameters to K = max_components, and any
protocol can leave dead components in the best CV fit. A zero-weight
component can never regain weight under EM, so such components are
dropped here (the same masking FittedGMM applies). Otherwise the
bootstrap, refit and MC stages would carry them, and they would shrink
the auto-sized chunks by the padding factor.

### `_bootstrap_counts`

The indices are floor(u N) from float64 uniforms, uniform to <= N/2^52;
the clamp guards only u == 1.0 (no float64 u < 1 reaches N). The uniform
and index vectors are temporaries, not bound to names: binding them
across iterations keeps two extra (N,) vectors live (24 B/sample instead
of 16).

### `estimate`, float64 entry

The entry pipeline (finiteness check, rank transform, affine
standardisation) runs at float64 BEFORE the cast to the working dtype:
casting first quantises large-offset columns (MJDs, Julian dates,
timestamps) at float32 resolution before the centring that would have
made them representable, silently destroying the estimate. Integer and
half-precision inputs are promoted the same way; already-float32 input
cannot be un-quantised, so the upcast there is best effort. The one-off
float64 copy of X at entry is the price.

### `estimate`, standardisation

Internal per-variable affine standardisation.  MI, cMI and TC are
exactly invariant under per-variable affine maps, and the KL quantities
under a map shared by both columns, so no estimated quantity changes;
it makes `reg_covar` a scale-free floor and prevents float32 Cholesky
failures on unstandardised, strongly correlated pairs.  The fitted
model returned by `return_fit` is mapped back to the original units.

### `estimate`, return_fit units

The covariance is factorised, and repaired if necessary, in standardised
units, where reg_covar, the absolute dead test and the repair bump scale
are meaningful. Both the covariance and its factor are then transformed
back at float64: D L is lower-triangular and (D L)(D L)^T = D cov D
exactly, so cov == chol chol^T survives the change of units, and a
float32 working dtype cannot overflow (or zero out) the original-units
covariance of a healthy fit.

### `_mc_stage_categorical`

log sum_c' pi_c' p(x|c') is accumulated as a running logaddexp over the
categories, so the peak holds three (b,S) tensors instead of a (C,b,S)
stack plus its C-element list. The results differ from the stacked
logsumexp at floating-point roundoff only.

### `estimate_categorical`, labels

Non-numeric labels (strings and objects, the common case for categorical
astrophysical data) are factorised via numpy. The categories are the
sorted unique labels, as np.unique orders them; MI only needs the
partition.

### `estimate_categorical`, seeds

One coincidence of streams is known and of low risk: category ci+10's
init stream at K=30 equals category ci's bootstrap stream
(997*10 + 30 = 10_000). It needs >= 11 categories AND max_components >= 30
(default 10), and the streams drive different data, so it is documented
rather than re-seeded under the validated init/bootstrap offsets.
