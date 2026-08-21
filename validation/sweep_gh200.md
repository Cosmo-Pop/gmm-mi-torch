# gmm-mi-torch sweep: NVIDIA GH200 120GB

```
{
 "torch": "2.11.0+cu128",
 "python": "3.11.7",
 "gmm_mi_torch": "n/a",
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
| float32 | 8 | 0.0488 | 0.0022 | 18 | 21.8 | 5.72 |
| float32 | 32 | 0.0488 | 0.0022 | 18 | 9.5 | 5.72 |
| float32 | 128 | 0.0488 | 0.0022 | 18 | 6.5 | 5.72 |
| float32 | 300 | 0.0488 | 0.0022 | 18 | 5.6 | 5.72 |
| float32 | auto | 0.0488 | 0.0022 | 18 | 5.6 | 5.72 |
| float64 | 8 | 0.0486 | 0.0021 | 19 | 16.7 | 11.41 |
| float64 | 32 | 0.0486 | 0.0021 | 19 | 9.9 | 11.41 |
| float64 | 128 | 0.0486 | 0.0021 | 19 | 8.1 | 11.41 |
| float64 | 300 | 0.0486 | 0.0021 | 19 | 7.6 | 11.41 |
| float64 | auto | 0.0486 | 0.0021 | 19 | 7.6 | 11.41 |

## one EM step: ms per replicate-iteration vs batch size

| dtype | Ra=1 | Ra=8 | Ra=32 | Ra=64 | Ra=128 | Ra=300 | Ra=600 | Ra=1200 |
|---|---|---|---|---|---|---|---|---|
| float32 | 1.127 | 0.249 | 0.109 | 0.093 | 0.082 | 0.075 | 0.073 | 0.073 |
| float64 | 1.049 | 0.193 | 0.128 | 0.118 | 0.114 | 0.113 | 0.114 | 0.117 |
