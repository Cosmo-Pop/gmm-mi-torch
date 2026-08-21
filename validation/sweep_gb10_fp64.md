# gmm-mi-torch sweep: NVIDIA GB10

```
{
 "torch": "2.11.0+cu128",
 "python": "3.12.3",
 "gmm_mi_torch": "n/a",
 "device": "cuda",
 "matmul_precision": "highest",
 "allow_tf32": false,
 "gpu": "NVIDIA GB10",
 "sms": 48,
 "cc": "12.1",
 "total_gb": 130.6,
 "free_gb": 84.0
}
```

## estimate(): hard pair, tester settings

| dtype | em_chunk | MI | sigma | k | wall (s) | peak (GB) |
|---|---|---|---|---|---|---|
| float64 | 300 | 0.0486 | 0.0023 | 20 | 1330.6 | 11.39 |
| float64 | auto | 0.0486 | 0.0023 | 20 | 1330.5 | 11.39 |

## one EM step: ms per replicate-iteration vs batch size

| dtype | Ra=1 | Ra=8 | Ra=32 | Ra=64 | Ra=128 | Ra=300 | Ra=600 | Ra=1200 |
|---|---|---|---|---|---|---|---|---|
| float64 | 41.963 | 27.513 | 21.419 | 20.815 | 20.502 | 20.296 | 20.224 | 20.192 |
