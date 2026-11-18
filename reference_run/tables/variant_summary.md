| Experiment | Variant | N | Test Acc (Mean ± SD) | 95% CI | Test Loss | Train Acc | Params (Mid / Total) | 1-Qubit Entropy (bits) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| e2_moons | Hybrid (entangled) | 20 | 0.9559 ± 0.0098 | [0.9516, 0.9599] | 0.1294 | 0.9668 | 36 / 58 | 0.8285 ± 0.0219 |
| e2_moons | Hybrid (no entanglement) | 20 | 0.9613 ± 0.0058 | [0.9588, 0.9638] | 0.1162 | 0.9733 | 36 / 58 | 0.0000 ± 0.0000 |
| e2_moons | Classical MLP | 20 | 0.9337 ± 0.0390 | [0.9165, 0.9498] | 0.1799 | 0.9407 | 40 / 62 | - |
| e2_moons | No middle block | 20 | 0.9147 ± 0.0464 | [0.8954, 0.9339] | 0.2032 | 0.9193 | 0 / 22 | - |
| e3_parity | Parity quantum (entangled) | 20 | 0.9996 ± 0.0008 | [0.9992, 0.9999] | 0.0023 | 1.0000 | 18 / 28 | 0.2384 ± 0.0556 |
| e3_parity | Parity quantum (entanglement-off) | 20 | 0.5127 ± 0.0213 | [0.5044, 0.5223] | 0.7022 | 0.5602 | 18 / 28 | 0.0000 ± 0.0000 |
| e3_parity | Classical MLP (reference) | 20 | 0.9982 ± 0.0015 | [0.9976, 0.9988] | 0.0071 | 1.0000 | 40 / 62 | - |
