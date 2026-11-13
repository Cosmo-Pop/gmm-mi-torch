# Changelog

## 0.11.0 (2026-11-13)

Documentation only; the code is that of 0.10.2.

- An API reference in `docs/`, one page per module (`estimator.md`,
  `em.md`, `selection.md`, `gaussian.md`, `plotting.md`), in the form the
  pop-cosmos documentation uses: for each public function, class and
  method, its signature, a link to its source line, and its parameters,
  returns, exceptions and warnings as tables. The pages are generated from
  the docstrings by `docs/make_api_docs.py` (install the new `docs` extra:
  `pip install -e '.[docs]'`); `--check` reports a page that is out of date. The README
  has a Documentation section listing them.
- The plotting helpers' docstrings are completed for the pages: one
  undocumented parameter (`create_heatmap`'s `fsize`), `**kwargs`, `str`
  types, and no "Returns / None" sections on functions that return
  nothing.
- Also since the 0.10.2 entry was written: the design notes' "Random
  streams" table names all six seed offsets (commit 31babc0).

## 0.10.2 (2026-11-13)

Readability tidy from an overall read of the code against the readability
proposal (scoping/review-overall-readability-2026-11-13.md, findings F1-F3).
No behaviour change: the same random streams, results and messages.

- The README now points to docs/estimate_notes.md, where `estimate`'s
  guidance on component selection, standardisation, preprocessing,
  initialisation, memory and reproducibility lives.
- Two seed offsets written as bare numbers are named constants beside the
  others: `_BOOTSTRAP_SEED_OFFSET` (10_000, the bootstrap stream) and
  `_CATEGORY_SEED_STRIDE` (997, the per-category seed of
  `estimate_categorical`). Same values.
- Docstrings: eleven small numpydoc style fixes, now checked on every
  release, and one summary that was repeated. One section banner corrected.

Test matrix (passed / skipped, 133 tests): CPU fp64 131 / 2, CPU fp64
legacy 129 / 4, CUDA fp32 132 / 1, CUDA fp32 legacy 130 / 3.

## 0.10.1 (2026-11-13)

Tests only; the package is identical to 0.10.0. Seven tests that patch a
module setting or swap in a stand-in to reach a code path now also check
that the patched path really ran (commit a1157a0; note N6 of the phase 3
landed read, scoping/review-findings-readability-phase3-2026-11-11.md):
their assertions held on the default path too, so a patch that stopped
reaching its target would have gone unnoticed. Disabling each patch makes
its test fail at the new check. Test matrix (passed / skipped, 133 tests):
CPU fp64 131 / 2, CPU fp64 legacy 129 / 4, CUDA fp32 132 / 1, CUDA fp32
legacy 130 / 3.

## 0.10.0 (2026-11-11)

Readability, phase 3 of scoping/readability-proposal-2026-11-04.md (items
3.6, 3.7 and 3.8, with the owner's rulings of 2026-11-11 for 3.7). Names,
layout and documentation only, no behaviour change: no formula, constant,
default, random stream, error or warning text moved, and every public
function keeps its name, arguments and results.

- The package is split into modules (3.7): `constants`, `precision`,
  `chunking`, `gaussian`, `em`, `selection` and `estimator`, one stage
  each, every statement moved verbatim. `gmm_mi_torch.estimate`,
  `estimate_categorical` and `FittedGMM` are where they were, and
  `gmm_mi_torch.estimator` still provides every public name it defined
  (`em_fit`, `val_ll`, `select_batched`, `select_sequential`, the Gaussian
  helpers, `LOG2PI`, `EMResult`, `ValLLResult`). Private (underscore)
  names now live in, and must be imported from, the module that defines
  them; a script that assigned one of them on `gmm_mi_torch.estimator`
  at run time (for example `_EM_KERNEL`) now changes nothing, because
  each module holds its own binding (the supported kernel switch, the
  `GMM_MI_TORCH_EM_KERNEL` environment variable, is unchanged). Any
  warning attributed to a frame inside the package now names that frame's
  module (for example `gmm_mi_torch.em` or `gmm_mi_torch.selection`, not
  `gmm_mi_torch.estimator`); text, category and count are unchanged, so
  only a warnings filter keyed on the module name is affected.
- Long comment blocks move to the design notes (3.6): the 24 blocks of
  more than three lines are in docs/design_notes.md, "Notes moved from the
  code", verbatim; the code keeps one line per block naming what follows
  and the note.
- Docstrings (3.8): every function, class and public method has numpydoc
  Parameters and Returns sections (two constructors are documented on
  their classes; a few dunder methods and inner wrappers have none). `estimate`'s Notes move verbatim to
  docs/estimate_notes.md, so `help(estimate)` is 100 lines shorter.
- Also in this release: inside the EM step kernels the weight accessor
  and the active-replicate count are named `weight_chunks` and `n_active`,
  as at their call sites (commit f35a213, after v0.9.0); and, after the
  landed read, a docstring tidy (commit 23dd559): the one-line summaries
  added in 3.8 no longer repeat the sentence after them, grouped
  Parameters entries with defaults say ", optional", and `weight_chunks`
  in two EM helpers is described as used only on the streamed path.

Proof of no behaviour change, at each commit: for 3.6 the syntax tree
with docstrings is identical and every moved block is found verbatim in
the notes; for 3.7 every statement is identical in its new module, every
name resolves, the import-time kernel switch behaves as before, and the
earlier releases (v0.7.2, v0.8.0 and, in the landed read, v0.9.0) and the
new code agree bitwise on CPU and CUDA with both kernels;
for 3.8 the syntax tree without docstrings is identical, every lost word
of an old docstring is on an exact list, and `numpydoc.validate` passes
its structure and completeness checks on all 84 objects. The landed read
by a session that did not write the code
(scoping/review-findings-readability-phase3-2026-11-11.md, verdict TAG)
confirmed this with its own syntax-tree, binding and bitwise checks. The v0.7.0
reference sets compare with 0 mismatches on both kernels and the seven
public signatures equal v0.9.0's at every commit. Test matrix (passed /
skipped, 133 tests): CPU fp64 131 / 2, CPU fp64 legacy 129 / 4, CUDA fp32
132 / 1, CUDA fp32 legacy 130 / 3 at every commit. The walkthrough
notebook and the multi-GPU example re-run clean.

## 0.9.0 (2026-11-11)

Readability, phase 2 of scoping/readability-proposal-2026-11-04.md (items
3.3 and 3.5, with the RSE's ruling of 2026-11-06 for 3.5). Names and layout
only, no behaviour change: no formula, constant, default, random stream,
error or warning text moved, and every public function keeps its name,
arguments and results.

- The EM iteration untangled (3.3): the loop of `_em_fit_core` now reads as
  its steps, with each former branch body in a function of its own:
  `_check_reroute` (the kernel-reroute check, GEMM kernel), `_em_iteration`
  (one EM iteration, dispatching to the existing step kernels),
  `_em_step_mixed` (a batch split between the GEMM and legacy kernels) and
  `_compact_converged` (store the converged replicates, drop them from the
  batch). The arithmetic lines moved verbatim, in the same order.
- The threaded hyperparameters as two plain dicts (3.5): `estimate` and
  `estimate_categorical` build `selection` (protocol,
  fixed_components_number, max_components, n_folds, n_inits,
  metric_method, threshold_components, patience) and `fit` (reg_covar,
  threshold_fit, max_iter, em_chunk, n_chunk, init_type, scale, seed) once;
  `_select_and_fit(X, selection, fit)` (formerly 17 positional arguments)
  and `_cv_fit_for_k(X, K, n_folds, n_inits, fit, init_chunk=None)`
  (formerly 13) read them. The keyword arguments of every public function
  are unchanged; code that called the two private helpers directly must
  pass the dicts.

Proof of no behaviour change, at each of the two commits: the syntax tree
of `estimator.py` with docstrings removed matches v0.8.0 piecewise (each
new function against the block it came from, the dict reads mapped back to
the old names), with every remaining difference on an exact allow-list and
a planted-edit negative test; the v0.7.0 reference sets compare with 0
mismatches on both kernels; the earlier releases (v0.7.2, which is bitwise
v0.8.0, and v0.8.0 itself) and the new code, run side by side and
compared with `torch.equal` on CPU and CUDA with both kernels (`em_fit`
with mixed and all-legacy batches, `val_ll`, the selection protocols,
`_init_params`, `estimate`, `estimate_categorical`, and every warning's
text), agree with 0 mismatches; `inspect.signature` of the seven public
callables equals v0.8.0's. Test matrix (passed / skipped, 133 tests): CPU
fp64 131 / 2, CPU fp64 legacy 129 / 4, CUDA fp32 132 / 1, CUDA fp32
legacy 130 / 3 at both commits. The walkthrough notebook and the
multi-GPU example re-run clean.

The landed read by a session that did not write the code
(scoping/review-findings-readability-phase2-2026-11-10.md, verdict TAG)
checked the commits with its own syntax-tree script, recorded every
argument reaching the fits through `estimate` and `estimate_categorical`
with all sixteen hyperparameters set away from their defaults (old and new
records equal), and compared v0.8.0 with the new code bitwise on CUDA with
both kernels (0 mismatches). It found no behaviour change; the one claim it
corrected (peak memory moves by a few kilobytes in either direction, not
"can only fall") is fixed in the pass's record.

## 0.8.0 (2026-11-09)

Readability, phase 1 of scoping/readability-proposal-2026-11-04.md (items
3.1, 3.2 and 3.4, as approved by the RSE on 2026-11-06). Names and layout
only, no behaviour change: no formula, constant, default, random stream,
error or warning text moved, and every public function keeps its name,
arguments and results.

- Named results (3.1): `_em_fit_core` returns an `EMResult` (w, mu, cov,
  ll, n_unconverged, n_rerouted), `_val_ll_core` a `ValLLResult` (values,
  n_rerouted), and `estimate_categorical` keeps its per-category fits as
  `CategoryFit` (w, mu, cov, chol), all `typing.NamedTuple`s; the
  positional reads (`o[4]`, `o[5]`, `fits[cj][3]`, ...) are field reads.
  `em_fit` and `val_ll` still return plain tuples / tensors.
- Bookkeeping names (3.2): inside `_em_fit_core` and `_val_ll_core`,
  `ai`, `sa`, `leg`, `gi`/`li`, `Ra`, `nl`, `act`, `wc`, `w_n`/`mu_n`/
  `cov_n` are `active_idx`, `weights_active`, `legacy_mask`,
  `gemm_idx`/`legacy_idx`, `n_active`, `legacy_sample_chunk`,
  `active_rows`, `weight_chunks`, `w_next`/`mu_next`/`cov_next`.
- Hoisted chunk callbacks (3.4): the closures inside `_init_params`,
  `estimate` and `estimate_categorical` are module-level functions with
  explicit arguments, bound with `functools.partial`:
  `_init_mstep_random`, `_draw_init_blocks`,
  `_normalise_responsibilities`, `_kmeans_assign_stats`,
  `_hard_assignments`, `_finish_kmeans_params`, `_mc_stage`,
  `_mc_stage_categorical`, each with a numpydoc docstring and a direct
  unit test (8 new tests; 133 in all).
- Two leftovers tidied after the landed read (commit f6f90cd): a
  continuation line in `_val_ll_core` re-wrapped, and in
  `_init_mstep_random` the per-replicate generator list built inside the
  single-pass branch, as in v0.7.2's closure, so the streamed path no
  longer builds a list it discards.

Proof of no behaviour change, at each of the three item commits: the syntax
tree of `estimator.py` with docstrings removed matches v0.7.2 under the
allow-listed renames and moves (the pass's `astcheck_080.py`, kept with
the gate logs outside the repository), the
v0.7.0 reference sets compare with 0 mismatches on both kernels, and
`inspect.signature` of `estimate`, `estimate_categorical`,
`FittedGMM.__init__`, `em_fit`, `val_ll`, `select_batched` and
`select_sequential` equals v0.7.2's. Test matrix (passed / skipped): CPU
fp64 123 / 2, CPU fp64 legacy 121 / 4, CUDA fp32 124 / 1, CUDA fp32
legacy 122 / 3 at the first two commits (125 tests); 131 / 2, 129 / 4,
132 / 1, 130 / 3 at the third (133 tests). The walkthrough notebook and
the multi-GPU example re-run clean.

The landed read by a session that did not write the code
(scoping/review-findings-readability-phase1-2026-11-09.md, verdict TAG)
added a check the gates above lack on CUDA: v0.7.2 and the new code run
side by side in one process and compared with `torch.equal`, on CPU and
CUDA with both kernels (`_init_params`, `estimate` and
`estimate_categorical`, float32 and float64, single-pass and streamed):
0 mismatches. At the tidy commit the same comparison, the reference sets
and the signatures are again clean, the syntax tree differs from the
third commit's by the one moved statement, and the matrix is again
131 / 2, 129 / 4, 132 / 1, 130 / 3.

## 0.7.2 (2026-10-25)

A small patch under the owner's lightweight process
(scoping/design-report-v0.7.2-2026-10-25.md; the starting session reads
the landed diff). No random stream moves: valid input runs bitwise as
at 0.7.1 (the v0.7.0 reference sets compare clean on both kernels).

- Behaviour change: `mc_chunk` is validated at entry in `estimate` and
  `estimate_categorical`. Its type/sign check (2.5 -> TypeError; 'bad'
  or 0 -> ValueError, naming the argument) used to run only after
  component selection — in `estimate_categorical` after every
  category's selection and bootstrap EM — so a malformed value cost a
  full selection before it raised. It now raises from the shared entry
  validation, before any fitting, which makes the 0.6.1 entry's "fail
  at entry" claim true. The 'auto' sizing still happens after
  selection (it needs the selected k) and receives the caller's value
  unchanged. New test: test_mc_chunk_validated_at_entry (zero
  _select_and_fit calls on all six bad inputs; it fails on 0.7.1).
- Docs: `estimate` gains a Warns section listing every RuntimeWarning
  its path can emit, with its condition and frequency (per call, per
  skipped k, per em_fit or scoring call, per out-of-memory halving, per
  covariance factorisation); `estimate_categorical`'s Warns refers to it
  per category; `select_batched` and `select_sequential` gain Raises.
  The Raises sections of both estimators list mc_chunk at entry.

Test matrix at the code commit 0069918: 123 passed / 2 skipped (CPU
fp64), 124 / 1 (CUDA fp32), 121 / 4 (CPU fp64 legacy), 122 / 3 (CUDA
fp32 legacy); 125 tests.

## 0.7.1 (2026-10-24)

Documentation only, no behaviour change: `gmm_mi_torch/estimator.py`
parses to the same syntax tree as at 0.7.0 once docstrings are removed,
the v0.7.0 reference sets compare clean on both kernels, and the test
matrix is unchanged. Built under report-before-code
(scoping/design-report-v0.7.1-docs-2026-10-24.md, ruled in
design-review-v0.7.1-docs-2026-10-24.md) and read as a landed diff before
the tag.

- `estimate`: each parameter entry is two to four lines; the longer
  explanations moved, every fact kept, into its Notes under short
  headings (selection; standardisation and regularisation;
  preprocessing; initialisation; chunking and memory; reproducibility).
- Raises sections on `estimate`, `estimate_categorical`, `em_fit` and
  `val_ll`, listing the conditions the code raises; Warns sections on
  `em_fit` and `val_ll`.
- `estimate_categorical`, `em_fit`, `select_batched`, `select_sequential`
  and `FittedGMM` document every parameter in a form numpydoc parses
  (grouped "As in `estimate`" entries replace a free paragraph and
  line-wrapped group headers).
- Private helpers state purpose, contract and invariants only: dated
  sentences, review references and "until vX.Y" history moved to the new
  docs/design_notes.md (mechanisms and measurements: architecture, kernel
  reroute, memory models, bootstrap resample bounds, random streams, CUDA
  reproducibility, the precision pin) or were already recorded in this
  changelog.
- Corrections to the documentation: 'TC' needs at least two columns (the
  docstring said any number); `scale` also sets the 'minmax'
  initialisation's standard deviation; `n_bootstrap` replicates are
  batched per em_chunk; a malformed `mc_chunk` is rejected after
  component selection (after every category's selection and bootstrap
  in `estimate_categorical`), not at entry; and every chunk-invariance
  claim (`em_fit`, `val_ll`, the initialisation, the out-of-memory
  retry, `mc_chunk`) now reads "bitwise on CPU at float64 (at float32
  also bitwise except em_chunk=1 with a one-component model, where the
  M-step's single-row tail product takes a different BLAS path: 3e-8 in
  a measured MI), reduction order on CUDA": on CUDA float64 is not
  bitwise chunk-invariant either (fitted
  means moved by 1e-14 to 1e-12 between em_chunk values in two measured
  cases; float32 by up to 7e-3), so CPU, not "float64 or CPU", is the
  bit-reproducible setting, and only the Monte-Carlo samples, not the
  results, are independent of mc_chunk on CUDA. Also: README's legacy
  kernels are no longer said to be available "for one release" (they
  still are), and `estimate_categorical` documents that its reg_covar
  floor is relative to the variance of X over all categories.
- README: the CUDA batching paragraph became a subsection of short
  bullets, a Known limitations section points to the design notes, and
  the kernel, large-N, multi-GPU and test paragraphs are split; every
  number is kept.

Test matrix at the docstring commit d081acb and at the first release
commit df6628f (the docs-only fix-first commit after it changes no code:
its estimator.py syntax tree is v0.7.0's): 122 passed / 2 skipped (CPU
fp64), 123 / 1 (CUDA fp32), 120 / 4 (CPU fp64 legacy), 121 / 3 (CUDA
fp32 legacy). Reference sets ref_v070 / ref_v070_legacy: 0 mismatches.

## 0.7.0 (2026-10-24)

One change, on one path: above torch.multinomial's 2^24-category cap the
bootstrap resample draws its indices as floor(u N) from float64 uniforms
instead of a torch.randint whose 32-bit draw, reduced modulo N, drew the
first 2^32 mod N indices ceil(2^32/N)/floor(2^32/N) times as often as
the rest (+2.4% on the first 95% of rows at N=1e8; +0.233% on the first
49.7% at N=1e7, 9-12 sigma at eight replicates). Built under
report-before-code (scoping/design-report-v0.7.0-2026-10-24.md, ruled in
design-review-v0.7.0-2026-10-24.md with ten amendments, all adopted) and
read as a landed diff before the tag. The minor-version bump follows the
project's precedent for random-stream changes: the N > 2^24 resample
stream moves on both devices (same seed derivation); results are
bitwise v0.6.1 on CPU everywhere else, and the N <= 2^24 multinomial
stream, CPU and CUDA, is unchanged.

- The new draw is uniform to <= N/2^53 on CPU (uniforms on the 2^-53
  grid) and <= N/2^52 on CUDA (the kernel's half-step values above 0.5
  round to even onto the 2^-52 grid, measured at torch 2.11): 2.2e-8 at
  N=1e8, no regional structure; the first-region/rest count ratio at
  N=1e7 sits within 2.2 sigma of 1 over three seeds on both devices.
  The clamp to N-1 guards only a kernel returning u = 1.0: for every
  float64 u < 1 and N <= 2^53, floor(fl(u N)) <= N-1 (the deficit
  N 2^-53 is at least half an ulp of N, equal only at powers of two,
  where the product is exact).
- Transient 16 B/sample per replicate at a time, measured, the same as
  the randint form's (whose docstring counted one of its two live index
  vectors); the draw is written as one expression so the uniforms and
  the indices are released before the next replicate (binding them
  measures 24 B/sample). At N=1e8 that is 1.6 GB, 8% of the float32
  counts at 50 replicates.
- Cost ~1.6x the randint draw on both devices (draw + scatter at N=3e6,
  ten replicates, best of three: CPU 0.14 -> 0.22 s, CUDA 12 -> 19 ms);
  at N=1e8 with 50 replicates about 50 s on CPU, against tens of minutes
  of bootstrap EM. The 0.6.1 design report's sentence that the float64
  draw was faster was wrong.
- Tests: the stream-and-transients test spies torch.rand (one float64
  row per replicate, whatever the working dtype) and torch.randint (the
  seeds only); a uniformity test at N=1e7 (4-sigma gate; the randint
  draw fails it at 9-12 sigma) and a mechanism test with torch.rand
  replaced by fixed vectors (u = 1.0 clamped; nextafter(1, 0) and
  1 - 2^-52 floor to N-1 at N=16 and at the non-power-of-two N=12) are
  new; 124 tests. Matrix at the draw commit (passed/skipped): cpu64
  122/2, cuda32 123/1, cpu64_legacy 119/4 and cuda32_legacy 120/3, each
  legacy leg with one failure, test_collapsed_component_offdiagonal_bound,
  which is pre-existing (it fails identically on v0.6.1's code: its
  repair-warning assertion, added in the v0.6.1 fix-first commit, was
  verified on the default leg only, and the legacy centred kernel
  completes that case with nothing to repair). A test-only commit gates
  that assertion on the GEMM kernel; the test then passes on all four
  legs.
- Decided and not adopted, by the owner (2026-10-24): the CUDA
  multinomial keeps its beta-validated float64 stream for N <= 2^24 (the
  documented float32-resolution weight scatter, sd 0.02 at N=1e6 to
  ~0.24 near 1.1e7, stands; a device-dependent category cap was not
  adopted); the streamed CV-mask em_fit memory model does not carry the
  fixed cuBLAS-workspace term (the bench's caption on those rows stands).
  The items listed at 0.6.1 for a later pass are unchanged.
- Docs: README's reproducibility paragraph, docs/make_validation_summary.py
  and the regenerated docs/validation_summary.pdf describe the new draw.

## 0.6.1 (2026-10-23)

The pass for the 19 low findings and 3 sweep extras of the v0.5.0
closing review and the 3 docstring items of the v0.6.0 landed read,
built under report-before-code (scoping/design-report-v0.6.1-2026-10-23.md,
ruled in design-review-v0.6.1-2026-10-23.md; all amendments adopted) and
read as a landed diff before the tag. No random stream moves: results
are bitwise v0.6.0 on CPU everywhere except the repair of components
whose diagonal clamped (none in the reference set), and CUDA auto-sized
batches change only in the tight regime and in capped processes (the
accepted reduction-order class).

- Entry validation: n_chunk, threshold_fit, patience, em_chunk and
  mc_chunk fail at entry under their own names (a float n_chunk died
  inside the first streamed EM step; em_chunk=2.7 ran as 2 and
  mc_chunk=1.5 as 1; threshold_fit <= 0 or NaN turned every fit into a
  max_iter run); device accepts only 'cpu' and 'cuda[:i]' (other
  backends failed deep inside the fit or not at all); None/NaN/pd.NA
  among string labels raise the readable missing-label error;
  FittedGMM.score_samples takes the DataFrame it was fitted on and
  reports the caller's shape. New ValueErrors/TypeErrors on input that
  previously ran or failed late.
- One non-finite replicate metric no longer poisons selection: 'valid'
  ranks the finite k only (it was an IndexError), aic/bic mask the
  criterion and pick the best finite init (a NaN k was selected
  silently), the sequential search skips a non-finite k with a warning
  without consuming patience (it could end the search early).
- The init hopeless-case guard runs whatever em_chunk is, quotes the
  memory model's estimate, names n_chunk first and no longer recommends
  'minmax' to a 'minmax' run; the n_chunk docstring states the init
  chunk's block rounding.
- Auto-chunking: the 32-replicate floor is retired (it only bound when
  memory was tight, exactly where it over-requested; its fixed-term-blind
  cap still opened with a certain OOM); batches honour
  torch.cuda.set_per_process_memory_fraction; the MC model's slope is
  3d+2 per component (the d=1 categorical stage was 12% over at K=20);
  the features are assembled in place (108 -> 80 units at d=8, model
  unchanged) and the streamed paths release the previous chunk's
  features before the next (1.34x the model at R=1 -> 0.88x); the entry
  pipeline standardises in place on its private float64 copy (7.6 ->
  5.9 float64 copies of X at the peak of a CUDA-tensor run). The memory
  bench runs at fp32 and (--fp64) fp64 with val_ll, d=1 MC and small-R
  rows: 207 rows per dtype, every row at or below its model except the
  streamed CV-mask em_fit rows at N=4e5, d=2, where a fixed ~10-19 MiB
  cuBLAS workspace shows against 5-45 MiB models (1.06x at K=3 fp32,
  1.42x at K=20 with n_chunk=4096 at both dtypes) — an absolute term the
  budget fraction absorbs; whether the model should carry it as an
  additive term is a 0.7.0 question.
- Collapsed components: the M-steps bound the off-diagonals of a
  component whose diagonal was clamped to the reg_covar floor, so a
  duplicated-point fit at float32 with reg_covar ~1e-10 completes with
  repair warnings instead of aborting the whole estimate; _chol's error
  names the remedy.
- Documented limitations (quantified by the independent review, streams
  unchanged): the CUDA multinomial's float32-resolution per-sample
  weight scatter (sd 0.02 at N=1e6 to ~0.24 near 1.1e7, no regional
  tilt); the N > 2^24 randint path's 32-bit modulo tilt (+2.4% on the
  first 95% of rows at N=1e8; its 2^53-resolution replacement is held
  for 0.7.0 together with the owner's decision on the CUDA multinomial);
  the CPU 2^32 per-replicate seed space; and that on CUDA at float32 the
  replicate chunk, including an out-of-memory retry, can move the
  selected k on knife-edge data (float64 or CPU is bit-reproducible).
- Docs: the memory-model tooling index, the 0.5.0 mask-cost figure, the
  nine CUDA-gated tests, and the walkthrough's dtype advice.

Test matrix at the tag: 120 passed / 2 skipped (CPU fp64), 121 / 1 (CUDA fp32), 118 / 4 (CPU fp64 legacy), 119 / 3 (CUDA fp32 legacy).

## 0.6.0 (2026-10-23)

The fix pass for the six medium findings of the closing adversarial
review of v0.5.0 (scoping/review-findings-2026-10-23.md), built under
report-before-code (scoping/design-report-v0.5.1-2026-10-23.md, ruled in
design-review-v0.6.0-2026-10-23.md; all amendments adopted) and read as
a landed diff by the independent reviewer before the tag. The
minor-version bump follows the project's precedent for random-stream
changes: the float32 bootstrap resample and the CUDA default-init stream
move (both corrections; see the first two items). Everything single-pass
and float64 is bitwise v0.5.0 on CPU; the streamed (n_chunk) path moves
at reduction order for real-valued weights passed directly to em_fit or
val_ll (the module's own 0/1 masks and resample counts sum exactly), and
rerouted replicates' val_ll moves at float32 reduction order.

Test matrix at the tag: 114 passed / 2 skipped (CPU fp64), 115 / 1
(CUDA fp32), 112 / 4 (CPU fp64 legacy), 113 / 3 (CUDA fp32 legacy).

- The bootstrap probability row is float64 whatever the working dtype.
  torch's CPU multinomial accumulates the row's CDF sequentially in the
  row's dtype, and at float32 the step 1/N rounded differently in each
  binade of the running sum: a fixed, replicate-independent tilt of the
  sampling weights by index region (16 sigma at N=1e6; 33% max/min
  predicted at N=1e7) — a bias on index-ordered data the errorbar could
  not see. The float32 resample is now uniform and bitwise the float64
  one on CPU and CUDA; the float64 stream (the beta-validated one) is
  untouched, so every float32 run with a bootstrap moves at
  bootstrap-noise level. torch.multinomial's internal CUDA transient
  grows from 5.1x to 8.3x the float32 weights (outside the OOM guard;
  material only near the 2^24 cap with large n_bootstrap on small GPUs).
- The default and k-means initialisations stream with n_chunk, in two
  passes (means, then centred second moments, float64 accumulators), and
  the default init draws each replicate's responsibilities in fixed
  blocks of 2^14 samples from the replicate's generator on both paths,
  so the draw is invariant to em_chunk and n_chunk by construction on
  every device. v0.5.0 ran the init over the full N per replicate
  regardless of n_chunk (binding on memory 50-5700x before EM at the
  README's shapes and dying in an uncaught OutOfMemoryError at chunk 1)
  and its init_type docstring claimed the opposite; the caveat is back
  in its correct form, and the hopeless case (no n_chunk at huge N on
  CUDA) raises a RuntimeError naming n_chunk and init_type='minmax'.
  Behaviour: the CPU init draw is bitwise v0.5.0 (sequential blocks ==
  one shot); the CUDA default-init stream changes once more (init-noise
  level). The init model is (4d+3)K+12 units of the streamed sample
  chunk (GB10: at or below 0.97 of budget over d=1,2,8, K=1,3,20).
- The streamed EM and scoring paths read the per-replicate weights
  through an accessor one sample block at a time: no (R,N) copies for
  the expand()-view CV masks, no compaction copy when replicates
  converge, no unconditional s_val[gi] in val_ll. Measured (GB10, fp32,
  N=4e5, R=32, n_chunk=4096): em_fit 18x -> 1.06x and val_ll 21x -> 0.95x
  of the model at K=3 (K=1: 0.95x / 0.68x). The one remaining N-sized
  tensor per stage is the caller's own weights, held once.
- val_ll scores rerouted (collapsed) replicates with the legacy kernels
  in the same model-derived sample chunk em_fit streams them with: the
  CV scoring stage peaked 3.1x (d=2, K=20) to 9.1x (d=8, K=20) the
  budget em_chunk was sized with; now 0.28-0.99x on every measured row.
  Rerouted replicates' val_ll moves at float32 reduction order.
- The precision pin restores inherited fp32_precision knobs as
  inherited (torch >= 2.9): it wrote resolved values back explicitly,
  so a caller who had set only the generic knob left estimate() with
  the cuda/mkldnn matmul knobs pinned and their later generic change
  silently ignored. The one case the rule cannot tell apart — a child
  set explicitly to its parent's value — comes back inherited.
- metric_method='aic'/'bic' with device='cpu' no longer fails under
  torch.set_default_device('cuda') (the module's last unpinned factory).

## 0.5.0 (2026-10-21)

The hardening round that follows the v0.4.0 design pass: the deferred
memory pass (item 6 of the 2026-10-20 ruling, in the reviewer's order),
the rulings' loose ends, a re-executed walkthrough and a closing
adversarial review (scoping/, dated 2026-10-21). Statistics are
unchanged: every kernel-level result, the minmax/k-means selections, the
bootstrap resample and the MC stage are bitwise those of v0.4.0 on CPU
(a 36-entry reference comparison on both kernels). The minor-version
bump is for the two random-stream changes below, which shift fixed-seed
results at init-noise level, and for the new entry errors.

- Memory, selection stage: the default initialisation draws its (R,N,K)
  responsibilities per replicate chunk inside the out-of-memory guard,
  in place, each replicate from its own generator seeded off the
  caller's stream (chunk- and retry-invariant by construction on CPU and
  CUDA; chunked draws from one CUDA generator are not). Behaviour
  change: the default init's stream differs from v0.4.0, so fixed-seed
  results with init_type='random_responsibilities' move at init-noise
  level ('minmax' and the k-means inits are bitwise). The two full-size
  transients of the old draw are gone (GB10, R=120, N=20,000, K=20:
  2.1x -> 1.3x the draw at em_chunk=15), and the init stage is sized
  from its own model, over the full N, instead of the 3x GEMM model
  (it was 3.4-11x over budget and retried before any EM ran).
- Memory, CV masks: select_batched held the masks repeated
  max_components times (2(K+1)*I*F*N*itemsize: ~2 kB/sample at float64
  defaults, 1 kB at float32; 40 GiB of the README's 77.5 GiB N=1e8 run). em_fit and
  val_ll now take weights with the replicate axis split over leading
  dims and gather each chunk's rows from an expand() view: 183 MiB ->
  3 MiB per mask at N=2e5.
- Memory, M-step: the blocked float32 moment GEMM broadcast the feature
  matrix once per replicate (q/K responsibility buffers: 23x at d=8,
  K=2; 44x at K=1). It is batched over blocks with (k, r) folded into
  the GEMM rows — one transient copy of the responsibilities, F read in
  place: 23 -> 1.3 buffers and 5.1 -> 0.3 ms at d=8, K=2; the README's
  flagship shape (K=20, d=8, N=2e5) 3.3 -> 1.3 buffers and 39 -> 12 ms
  per M-step. CPU results bitwise; CUDA at float32 reduction order
  (ruled on before the code landed; scoping/design-review-batch3-*).
- Memory models re-derived on the GB10 over K=1..20, d=1..8 (117 rows,
  all now at or below 98% of budget; validation/bench_mem_models.py):
  GEMM EM max(K+6, 2K+3) + 3Kq/512 units of N_eff x itemsize per
  replicate after the consumed responsibilities are released before the
  next E-step (the old 3K under-sized K=1 2x), legacy (4d+4)K+4, init
  (4d+3)K+4 over N, MC max((3d+1)K+d+7, d^2+4d+6) (gmm_sample's (S,d,d)
  gather was missing), a per-stage fixed term for the features, and the
  rerouted replicates' legacy sample chunk derived from the model with
  the partition's weights gathered per sample chunk (no more (R,N)
  copies per iteration). End to end at a mocked 1 GiB budget: peak
  1744 -> 498 MiB, same answer.
- Memory, bootstrap counts: the validated multinomial draw stays one
  call but reads an expanded probability row and scatters an expanded
  ones row (6.0 -> 5.1x the weights at float32; torch.multinomial's own
  index matrix is what remains); the N > 2^24 randint path draws one
  replicate at a time from per-replicate generators (4.0x -> 1.1-1.5x;
  at N=1e8, B=50 the 74.5 GiB transient becomes 0.8 GiB) — its stream
  changes (second behaviour change; it was never the validated one).
- Auto-chunking counts torch's cached-but-free blocks as free (every
  stage after the first was sized against free minus cache), never lets
  the 32-replicate floor exceed what fits, and releases the failed
  attempt's memory after the exception is cleared rather than inside
  the handler (where it was a no-op). estimate_categorical's MC stage
  accumulates a running logaddexp instead of a (C,b,S) stack (GB10:
  18 / 60 / 191 MiB at C=3/12/40 -> 14 MiB flat).
- Entry guards: a training fold with fewer than d+1 samples is a
  ValueError (N=2-3 used to return MI ~5-6 nats silently); below 100
  samples per dataset or category a RuntimeWarning names the
  finite-sample bias. `device` accepts 'cuda:1', torch.device objects
  and integers and rejects unparseable or absent devices at entry.
  FittedGMM.score_samples takes 1-D input (n points of a 1-D model, or
  one d-D point) and names the expected shape otherwise.
- plotting: repeated mass levels and coarse grids no longer crash
  contouring, a constant column fails with a readable error, list input
  works for the histogram helpers, the ellipse weight marker is drawn
  once (upstream drew it once per level).
- Notebook re-executed end to end on CPU float64 with this release.

## 0.4.0 (2026-10-20)

The design pass for the remaining 2026-10-20 review findings, built under
report-before-code: each change below was proposed in
scoping/design-report-v0.4.0-2026-10-20.md, ruled on by an independent
reviewer (scoping/design-review-v0.4.0-2026-10-20.md, all amendments
adopted), and landed with the regression tests the ruling asked for.
Intentional behaviour changes: gaussianise ties are randomised by default,
the Monte-Carlo streams are reseeded per replicate, float32 runs on
large-offset or spike-bearing data now return the float64 answer (they
were wrong before), and AIC/BIC values shift by up to 2*N*threshold_fit
(same selections in all tests).

- Float32 is now safe on collapsed components: the GEMM kernels' uncentred
  moments leave a rounding residue ~ u*P*|x|^2 that swamped a
  reg_covar-floored spike away from the origin (zero-inflated MI biased up
  to +60% at float32 under nothing but repair warnings; float64 kernel
  parity broken below reg_covar ~1e-13). em_fit/val_ll now detect such
  replicates on the parameters entering every E-step and finish them on
  the centred legacy kernels, per replicate and chunk-invariantly, with
  one aggregated RuntimeWarning. Measured on the GB10 at float32 defaults
  (N=20,000, independent reviewer's measurements): ordinary multimodal
  data reroutes a few per cent of the over-specified CV replicates (3 of
  120; 14 of 120 on CPU at N=6,000) at ~+10% wall-clock with MI unchanged
  (0.9065+-0.0051 vs float64 0.9052+-0.0052), so the reroute warning is
  routine at float32 during selection — it says so, and is not a data
  problem; 30%-zero-inflated data reroutes nearly every replicate and
  costs ~2.7x wall-clock (3.4 s vs 1.25 s — still ~3.4x faster than
  float64's 11.8 s), returning 0.9693+-0.0056 where the undetected
  kernels gave 0.9757+-0.0064 (float64: 0.9698+-0.0052); on worse seeds
  the undetected float32 bias reached +60%. The detector is deliberately
  blind to near-collinear components with O(1) variances, where the
  float32 GEMM kernels are accurate and the legacy kernels are the worse
  choice; that class is pinned by must-not-reroute tests and documented
  in the kernel section.
- gaussianise breaks ties with seeded random ranks by default
  (tie_rank='random'; 'average' remains available): the v0.3.0 average
  ranks mapped a tied block to one probit atom and fabricated ~0.24 nats
  between INDEPENDENT 50%-zero-inflated columns. Random tie-ranks leave
  the MI estimand exactly unchanged (the original values are recoverable
  from the tie-broken ones) and restore ~0 there. Measured caveat, stated
  in the tie warning: on DEPENDENT atoms (a shared zero mask, truth 0.949
  nats) the randomised transform is biased low (~0.74) while
  preprocess='none' lands at ~0.94 — prefer 'none' for heavily tied data.
  The tie warning also now ignores two-sample blocks (float32-native data
  acquires harmless birthday pairs at large N).
- The entry pipeline (finiteness check, rank transform, standardisation)
  runs at float64 BEFORE the cast to the working dtype: float64 data with
  offset/sigma >~ 1e6 (MJDs, timestamps) used to be quantised at float32
  before the centring could protect it — MI silently destroyed at offset
  1e8, a hard abort at timestamp scale. float32 now tracks float64 at
  every offset. Integer/half inputs are promoted; fp16/bf16 working dtypes
  are rejected at entry.
- return_fit: the covariance is factorised (and repaired) in the fit's
  standardised units and back-transformed together with its Cholesky
  factor at float64. Strain-scale (1e-21) data no longer comes back with
  identity covariances, mixed column scales no longer inflate a repaired
  small column ~100x, and 1e20-scale float32 fits no longer overflow to
  identity. FittedGMM attributes are float64 numpy.
- Reproducibility: each bootstrap replicate draws its MC samples from its
  own seeded stream, so results are independent of mc_chunk and of
  out-of-memory retries by construction (and the GB10 measures the
  per-replicate draws slightly faster than the batched one). Fixed-seed
  results are bitwise reproducible on CPU including with 'auto' chunks;
  on CUDA, em_chunk changes reduction order (documented). |seed| <= 2**62
  and n_bootstrap <= 2**20 are enforced so the stream offsets stay valid.
- The precision pin also suspends ambient torch.autocast (a bf16 region
  used to change both the estimate and the selected k); EM warns once per
  call when replicates exhaust max_iter unconverged; AIC/BIC score the
  refit parameters exactly (em_fit's log-likelihood lags its returned
  parameters by one M-step — now documented); val_ll accumulates at
  float64 across sample chunks like the EM kernels;
  torch.set_default_device('cuda') no longer breaks the CV masks.
- plotting.find_contours: the 1-sigma ellipse constant becomes chi2(2 dof)
  at 0.6827 (2.2958); the inherited 0.77 was the inverted quantile, whose
  ellipse encloses 32% of the mass (upstream GMM-MI has the same constant;
  default contour_levels=[2] unaffected). The reg_covar docstring states
  the KL pooled-variance exception.

## 0.3.1 (2026-10-20)

Mechanical fix release addressing the no-design-decision findings of the
2026-10-20 adversarial review of v0.3.0 (the larger items — the GEMM
kernels' origin-moment cancellation, the gaussianise tie policy on
zero-inflated data, the dtype-cast/standardisation ordering, FittedGMM's
factorisation units, MC chunk reproducibility semantics and the memory
models — are held for design decisions). Estimates for a given seed are
unchanged on valid input; the one reporting change is that `return_k` now
counts the live components of the returned model.

### Entry validation and input handling

- Non-finite input (NaN/inf) is rejected with a ValueError at both entry
  points. With `preprocess='gaussianise'`, NaN rows used to rank silently
  as the largest values and fabricate MI on independent data (0.33 nats
  measured where the truth is 0, no warning); on the 'none' path a single
  NaN poisoned every replicate through the sample-weight product, defeated
  the EM stop test (exactly max_iter 'non-finite covariance reset to
  identity' warnings per stage) and died in a path-dependent far-away
  IndexError. Non-finite category labels are rejected too: torch.unique
  made every NaN its own category, surfacing as a misleading n_folds
  error. `select_batched` now raises the sequential path's informative
  RuntimeError when no validation metric is finite, instead of an
  IndexError from the parsimony mask.
- Integer arguments (seed, n_bootstrap, mc_samples, max_components,
  n_folds, n_inits, max_iter, fixed_components_number) are coerced with
  operator.index at entry: numpy integer seeds — what `rng.integers`
  yields in seed sweeps — crashed `_cv_masks` with 'TypeError: an integer
  is required', and a float mc_samples/n_bootstrap crashed only after the
  selection and bootstrap EM were already paid for. Floors are enforced:
  max_iter >= 1 (0 silently returned the random init, every bootstrap
  replicate identical and the errorbar pure MC noise), reg_covar > 0
  (0 let collapsed components through to garbage estimates or
  factorisation failures), mc_samples/max_components/n_inits/
  fixed_components_number >= 1, n_bootstrap >= 0, threshold_components
  >= 0 (negative made the parsimony mask all-False -> IndexError).
- The protocol/metric_method compatibility rules run before the fixed-k
  shortcut (a typo'd metric_method ran silently when
  fixed_components_number was set, contradicting the 0.3.0 'raise at
  entry' claim), and the public `select_batched`/`select_sequential`
  validate fold feasibility as the estimators do (n_folds > N silently
  returned k=1 with every metric -inf).
- pandas DataFrames (and any `__array__` container) are accepted for X,
  as the 'array-like' docstring promised; read-only arrays are copied so
  torch does not warn. String/object category labels are factorised
  through numpy's unique (categories in sorted label order) instead of
  dying in torch.as_tensor with a raw TypeError, and the
  smallest-category fold error names the offending label.

### Precision pin, autograd, reporting

- `_full_precision` speaks torch >= 2.9's per-backend fp32_precision API.
  Once user code had set any of those knobs — the documented way to
  enable TF32 since 2.9 — the legacy getter raised RuntimeError ('mix of
  the legacy and new APIs') and every public entry point crashed, even on
  CPU. The pin now saves, sets to 'ieee' and restores the generic,
  cuda.matmul and mkldnn.matmul settings exactly (the legacy-setter
  restore used to rewrite the mkldnn backend for a caller who had only
  enabled CUDA TF32), falling back to the legacy get/set pair on older
  torch. A CUDA-gated test asserts a TF32-sensitive GEMM result inside
  em_fit under both APIs — the old test asserted a flag string and was
  structurally blind to both defects.
- `estimate` and `estimate_categorical` run under torch.no_grad() and
  detach the input after as_tensor: an input carrying requires_grad
  (model latents taken without .detach() — the primary use case) built
  and retained the EM autograd graph of every iteration (measured 5-30x
  peak CUDA memory, growing with iteration count) and return_fit=True
  then crashed in FittedGMM on .numpy(). em_fit and val_ll are unchanged:
  differentiating a log-likelihood with respect to the parameters remains
  possible.
- `return_k` reports the live component count of the model that is
  actually bootstrapped, MC-integrated and returned: the 0.3.0 dead-
  component slice could silently return a smaller model under the
  selected/requested k (fixed_components_number=6 returning a 5-component
  fit). A RuntimeWarning fires when a requested fixed_components_number
  loses components during the CV fit.
- The gaussianise tie warning and the return_fit warning attribute to the
  caller's frame instead of the decorator line inside estimator.py, so
  Python's once-per-location filtering distinguishes call sites and
  module-keyed warning filters match the caller.

### Packaging, tests, validation scripts

- `[test]` extra: numpy >= 2.0 (np.trapezoid is 2.0-only; three
  categorical tests failed on numpy 1.x) and scipy (two rank-probit
  oracle tests import scipy.stats, previously only in the [plot] extra),
  so the documented `pip install -e ".[test]" && pytest` passes in a
  fresh environment.
- The 8 bench scripts under validation/ crashed on 0.3.0's `_chol` tuple
  return before any measurement (only bench.py had been updated); all
  call sites unpack the tuple now and validation/README.md records the
  broken window.
- README: the test-count sentence no longer double-counts the
  offline-gated test and names both gated tests; the dependency sentence
  matches the [test] extra.
- test_mc_chunk_auto_within_mc_noise compared mc_chunk=8 against CPU
  'auto' (= 8): bitwise the same run, so it could never fail on the
  documented default pytest invocation. It now compares mc_chunk=3
  against 'auto', which differ on every leg of the test matrix.

## 0.3.0 (2026-10-20)

Fix release addressing the 2026-10-16 adversarial review. Three intentional
behaviour changes mean results for a given seed shift relative to v0.2.2
(all well within the validated errorbars on continuous, well-conditioned
data): gaussianise now average-ranks ties, CV folds now follow the seed,
and dead padding components are dropped before the bootstrap (which moves
estimates at Monte-Carlo noise level).

### Estimator correctness

- `preprocess='gaussianise'` breaks ties by average rank (the scipy
  `rankdata` convention) instead of positionally: a tied block used to be
  mapped to a monotone probit ramp in row order, fabricating MI whenever
  row order correlates with another variable (a sorted column next to a
  constant column returned 1.06 nats; the truth, and the new result, is 0).
  A RuntimeWarning now names columns with more than 1% tied values. Ranks
  and quantiles are computed at float64, where float32 integer ranks round
  above 2^24 samples. `return_fit` with gaussianise warns that the model
  lives in rank-probit space (the rank transform has no parametric
  inverse).
- `cov == chol @ chol^T` is now an invariant: `_chol` returns the repaired
  covariance and every caller writes it back. The Monte-Carlo stage used to
  sample the joint under the repaired factor while scoring marginals
  refactorised from the raw covariance — two different mixtures whenever a
  replicate needed repair. `FittedGMM` factorises once at construction, so
  `covariances_`, `sample` and `score_samples` agree; the repair warnings
  now report correct counts and the cumulative bump (the exhaustion message
  claimed "1e4 x mean variance"; the ceiling is ~1.11e3 x the largest
  absolute diagonal entry), and the non-finite-reset warning is no longer
  swallowed when escalation also runs.
- The legacy chunked M-step gets the v0.2.1 reg_covar diagonal clamp and
  float64 cross-chunk accumulators, matching the GEMM path: float32
  rounding of the shifted moments could push a collapsed component's
  variance below the floor, or negative.
- Sequential selection: the patience counter resets on improvement and the
  best-scoring k is returned — with patience=2 and improvements at k=1 and
  k=3, the old counter arithmetic returned k=2, discarding the better
  model; exhaustion returned max_components regardless of scores; an
  all-NaN metric crashed with KeyError. `patience` is now a parameter of
  `estimate` and `estimate_categorical`.
- The seed now controls the CV folds, as the docstring always claimed
  (they were hardcoded to seed 42); forwarded with a fixed offset so the
  fold stream stays decoupled from the init and bootstrap streams.
- An empty validation fold scores -inf instead of a spurious 0.0 that beat
  every real fit in the fold argmax; degenerate CV settings (n_folds < 2,
  n_folds > N, a category smaller than n_folds) are rejected up front with
  a clear ValueError, as are typo'd quantity/protocol/metric_method
  values, wrong column counts (previously bare asserts that vanish under
  python -O), bad n_chunk, a multi-column X passed to
  `estimate_categorical` (it was silently flattened), and a typo'd
  GMM_MI_TORCH_EM_KERNEL (it silently selected the legacy kernels).
- `estimate` and `estimate_categorical` share one selection->bootstrap
  pipeline (they had drifted): the categorical path no longer silently
  drops metric_method under protocol='sequential' (it raises, as estimate
  does), exposes `threshold_components` and `patience` instead of
  hardcoding them, and gains the N > 2^24 bootstrap fallback.
- `n_bootstrap=1` returns std=None instead of NaN (one replicate has no
  sample deviation).

### Memory and precision

- Dead padding components are sliced away after selection: the batched
  protocol pads to K=max_components, and the dead components used to ride
  through the bootstrap, refit and MC stages (~6.7x wasted work at
  k_best=3, max_components=20) while shrinking the auto-sized replicate
  batches by the same factor.
- `val_ll` scores replicates in `em_chunk` batches under the same
  halve-and-retry CUDA out-of-memory guard as `em_fit` (it used to
  evaluate all replicates in one unguarded allocation that could OOM an
  estimate whose EM stage had already fitted), and `_init_params` runs its
  replicate chunks under the same guard.
- `em_fit` and `val_ll` pin float32 matmuls to full precision at entry, so
  direct kernel calls no longer run under ambient TF32; the pin is
  process-global (not thread-local), now documented.

### Packaging, examples, docs

- New `plot` extra (`pip install ".[plot]"`): `gmm_mi_torch.plotting`
  imports numpy, matplotlib and scipy, none of which the base install
  guarantees.
- `examples/mi_grid_multigpu.py` (and the README copy): workers index into
  the inherited CUDA_VISIBLE_DEVICES mask instead of writing absolute GPU
  indices that escape a scheduler's allocation (e.g. Slurm), and the
  script exits cleanly on a machine with no visible GPU.
- Doc fixes: the em_chunk docstring documents the GEMM byte model (not
  just the legacy one); "Piras et al. 2022" -> 2023; README shell quoting
  for zsh; validation/README.md and gb10_report.md corrections; the three
  stale validation bench scripts import the package instead of the
  pre-package flat module they were silently timing.

## 0.2.2 (2026-10-16)

Documentation release; no code changes.

- README: optional fresh-environment steps, references for the GEMM
  formulation, self-contained multi-GPU example (also shipped as
  `examples/mi_grid_multigpu.py`; must be run as a script, not pasted into
  an interactive session).
- Benchmarks refreshed at v0.2.1: the README micro-benchmark re-measured
  (CUDA fp32 selection 0.90 -> 0.21 s on the GB10), and real-data beta
  numbers added (75-pair MI grid, N = 30,000: 53 min on 16 CPU cores vs
  7.5 min fp32 / 5.1 min fp64 on one A100).
- `docs/validation_summary.pdf` regenerated with a section covering
  v0.1.1 - v0.2.1 and the beta-test validation.

## 0.2.1 (2026-08-31)

- Fix an abort ("covariance factorisation failed even after diagonal
  regularisation") on float32 fits of zero-inflated variable pairs, found in
  beta testing. Two causes: float32 moment accumulation could round a
  collapsed component's variance below the `reg_covar` floor (the M-step now
  restores the floor on the diagonal), and the repair bump was scaled by the
  mean diagonal entry, which is negative for exactly the components that
  need repairing (now the largest absolute entry). Non-finite covariances
  are additionally reset to identity instead of blocking the repair loop.

## 0.2.0 (2026-08-29)

- The EM inner loop evaluates Gaussian log-densities and M-step moments as
  matrix products against per-sample quadratic features. Results are
  unchanged (float64 agreement with the previous kernels at reduction-order
  precision); wall time per EM iteration drops by roughly 3x on GPU and 2x
  on CPU. The previous kernels remain available for this release via
  `GMM_MI_TORCH_EM_KERNEL=legacy`.
- Inputs are standardised internally (per variable; the KL quantities share
  one map between the columns). Estimates are invariant, fitted models are
  returned in the original units, and `reg_covar` now acts relative to each
  variable's variance. This removes float32 factorisation failures on
  unstandardised, strongly correlated data.
- A covariance whose factorisation fails is repaired per component with
  escalating diagonal regularisation (and a RuntimeWarning) instead of
  aborting the fit.
- float32 matmul precision is pinned to 'highest' for the duration of each
  call, so TF32 settings made by other libraries do not affect results.
- `gmm_mi_torch.__version__` added.

## 0.1.1 (2026-08-21)

- `em_chunk` and `mc_chunk` default to `'auto'`: on CUDA each stage's
  replicate batch is sized from the free device memory (measured peak-memory
  model), with a floor of 32; on CPU the previous default of 8 is kept. A
  CUDA out-of-memory error halves the chunk and retries. Results are
  chunk-invariant.

## 0.1.0 (2026-07-09)

- Initial release: pure-torch port of GMM-MI with batched selection
  protocols, vectorised bootstrap, analytic marginals, MI / conditional MI /
  total correlation / marginal KL / categorical MI, chunked EM for large N,
  and plotting helpers.
