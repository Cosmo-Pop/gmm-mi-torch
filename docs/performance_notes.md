# Performance notes

Details behind the Performance section of the [README](../README.md): how the replicate batches are sized, what is reproducible on each device, and the largest N run so far.

## Replicate batches and reproducibility

- The replicate batches `em_chunk` and `mc_chunk` default to `'auto'`. On CPU `'auto'` is 8 replicates per batch.
- On CUDA each stage's batch is sized to fill half of the free device memory (torch's cached blocks included) from per-stage peak-memory models measured on the GB10 (`validation/bench_mem_models.py`; the models are tabulated in [docs/design_notes.md](design_notes.md)).
- The batch is capped by `torch.cuda.set_per_process_memory_fraction` when one is set. There is no minimum batch size.
- A CUDA out-of-memory error halves the batch and retries, with a `CUDA out of memory; retrying with chunk=...` RuntimeWarning.
- The Monte-Carlo samples are independent of `mc_chunk` by construction: each bootstrap replicate draws them from its own seeded stream, so out-of-memory retries cannot shift them either.
- Results are independent of `em_chunk` and `mc_chunk` bitwise on CPU at float64; at float32 on CPU also bitwise except `em_chunk=1` with a one-component model, where the M-step's single-row tail product takes a different BLAS path (3e-8 in a measured MI). On CUDA either chunk changes the matmul summation order, so same-seed runs agree only to floating-point rounding unless both are set to integers and no out-of-memory retry occurs (the retry halves the batch whatever the setting, so a run's batch can depend on the memory free at the time).
- At float32 that rounding can change the selected k on data where two values of k score almost equally (k=5 vs 6, MI 0.4741 vs 0.4745 against an errorbar of 0.008 in a measured example); at float64 the differences stay at the level set by the order of summation (1e-14 to 1e-12 in the fitted means in two measured cases). A CPU run is the bit-reproducible setting (with that one float32 exception).

## Large N

Sample-dimension chunking (`n_chunk`) extends the reachable scale to N=10⁸, d=8 (total correlation in 23 min / 77.5 GiB at fp32 on 128 GB unified memory, measured at v0.2.x; v0.5.0 removed the two allocations that set that run's peak, the construction of the bootstrap counts and the repeated CV masks, and the run has not been repeated at N=10⁸ since).
