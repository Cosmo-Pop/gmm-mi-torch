# gmm-mi-torch sweep: NVIDIA GH200 120GB

```
{
 "torch": "2.11.0+cu128",
 "em_kernel": "gemm",
 "python": "3.11.7",
 "gmm_mi_torch": "0.2.0",
 "device": "cuda",
 "matmul_precision": "highest",
 "allow_tf32": false,
 "gpu": "NVIDIA GH200 120GB",
 "sms": 132,
 "cc": "9.0",
 "total_gb": 102.0,
 "free_gb": 101.4
}
```

## estimate(): hard pair, tester settings

| dtype | em_chunk | MI | sigma | k | wall (s) | peak (GB) |
|---|---|---|---|---|---|---|
| float32 | 8 | 0.0487 | 0.0022 | 18 | 17.6 | 5.72 |
| float32 | 32 | 0.0487 | 0.0022 | 18 | 5.3 | 5.72 |
| float32 | 128 | 0.0487 | 0.0022 | 18 | 2.6 | 5.72 |
| float32 | 300 | 0.0487 | 0.0022 | 18 | 1.9 | 5.72 |
| float32 | auto | 0.0487 | 0.0022 | 18 | 1.9 | 5.72 |
| float64 | 8 | 0.0485 | 0.0022 | 19 | 14.3 | 11.41 |
| float64 | 32 | 0.0485 | 0.0022 | 19 | 5.1 | 11.41 |
| float64 | 128 | 0.0485 | 0.0022 | 19 | 2.6 | 11.41 |
| float64 | 300 | 0.0485 | 0.0022 | 19 | 2.1 | 11.41 |
| float64 | auto | 0.0485 | 0.0022 | 19 | 2.1 | 11.41 |

## one EM step: ms per replicate-iteration vs batch size

| dtype | Ra=1 | Ra=8 | Ra=32 | Ra=64 | Ra=128 | Ra=300 | Ra=600 | Ra=1200 |
|---|---|---|---|---|---|---|---|---|
| float32 | 1.581 | 0.203 | 0.050 | 0.030 | 0.023 | 0.019 | 0.017 | 0.016 |
| float64 | 1.328 | 0.166 | 0.049 | 0.034 | 0.026 | 0.023 | 0.021 | 0.020 |
