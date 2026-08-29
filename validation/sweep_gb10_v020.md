# gmm-mi-torch sweep: NVIDIA GB10

```
{
 "torch": "2.11.0+cu128",
 "em_kernel": "gemm",
 "python": "3.12.3",
 "gmm_mi_torch": "n/a",
 "device": "cuda",
 "matmul_precision": "highest",
 "allow_tf32": false,
 "gpu": "NVIDIA GB10",
 "sms": 48,
 "cc": "12.1",
 "total_gb": 130.6,
 "free_gb": 81.0
}
```

## estimate(): hard pair, tester settings

| dtype | em_chunk | MI | sigma | k | wall (s) | peak (GB) |
|---|---|---|---|---|---|---|
| float32 | 8 | 0.0486 | 0.0024 | 17 | 15.2 | 5.7 |
| float32 | 32 | 0.0486 | 0.0024 | 17 | 14.1 | 5.7 |
| float32 | 128 | 0.0486 | 0.0024 | 17 | 13.5 | 5.7 |
| float32 | 300 | 0.0483 | 0.0023 | 19 | 13.5 | 5.7 |
| float32 | auto | 0.0483 | 0.0023 | 19 | 13.5 | 5.7 |
| float64 | 8 | 0.0485 | 0.0023 | 15 | 391.4 | 11.39 |
| float64 | 32 | 0.0485 | 0.0023 | 15 | 144.2 | 11.39 |
| float64 | 128 | 0.0485 | 0.0023 | 15 | 100.7 | 11.39 |
| float64 | 300 | 0.0485 | 0.0023 | 15 | 91.6 | 11.39 |
| float64 | auto | 0.0485 | 0.0023 | 15 | 91.6 | 11.39 |

## one EM step: ms per replicate-iteration vs batch size

| dtype | Ra=1 | Ra=8 | Ra=32 | Ra=64 | Ra=128 | Ra=300 | Ra=600 | Ra=1200 |
|---|---|---|---|---|---|---|---|---|
| float32 | 0.563 | 0.213 | 0.220 | 0.210 | 0.206 | 0.204 | 0.199 | 0.197 |
| float64 | 0.704 | 5.223 | 1.564 | 1.558 | 1.251 | 1.243 | 1.177 | 1.143 |
