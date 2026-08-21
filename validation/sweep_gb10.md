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
| float32 | 8 | 0.0484 | 0.0023 | 19 | 72.9 | 5.7 |
| float32 | 32 | 0.0484 | 0.0023 | 19 | 51.7 | 5.7 |
| float32 | 128 | 0.0484 | 0.0023 | 19 | 47.3 | 5.7 |
| float32 | 300 | 0.0484 | 0.0023 | 19 | 46.1 | 5.7 |
| float32 | auto | 0.0484 | 0.0023 | 19 | 46.2 | 5.7 |

## one EM step: ms per replicate-iteration vs batch size

| dtype | Ra=1 | Ra=8 | Ra=32 | Ra=64 | Ra=128 | Ra=300 | Ra=600 | Ra=1200 |
|---|---|---|---|---|---|---|---|---|
| float32 | 3.558 | 0.979 | 0.732 | 0.720 | 0.706 | 0.694 | 0.683 | 0.678 |
