# CUDA Kernel Optimization Study for Softmax

### Naive → Reduction-Based → (Upcoming) Warp-Level Optimized

This repository explores how **CUDA kernel design** impacts performance across three fundamental operations — **Addition**, **Normalization**, and **Softmax**.

The goal is to progressively improve GPU efficiency through:

1. Naive implementations (baseline)
2. Reduction-based optimizations
3. (Next) Warp-level and fused-kernel designs

---

## Implementations

| Version               | Description                                                                     |
| :-------------------- | :------------------------------------------------------------------------------ |
| **Naive**             | Direct global memory access, atomic operations, multiple kernel launches.       |
| **Reduction-Based**   | Uses shared-memory reductions to reduce contention and improve throughput.      |
| **(Next) Warp-Level** | *Planned:* Fuse kernels and use warp shuffle intrinsics for maximum efficiency. |

---

## Comparison Summary

| Kernel            |                   Naive                  |                Reduction-Based               |         Next (Planned)        |
| :---------------- | :--------------------------------------: | :------------------------------------------: | :---------------------------: |
| **Addition**      | Very low efficiency (atomic bottlenecks) | ✅ Major speedup with shared-memory reduction |    🔜 Warp-level reduction    |
| **Normalization** |             Already efficient            |           ✅ Consistent performance           |   🔜 Fine-tune memory access  |
| **Softmax**       |         Multi-pass & memory-bound        |             ⚙️ Slight improvement            | 🔜 Fuse exp + sum + normalize |

---

## Key Takeaways

* **Atomic operations** severely limit performance — reduction-based design solves this.
* **Shared memory** drastically improves efficiency by reducing global memory traffic.
* **Kernel fusion** and **warp-level primitives** are the next logical steps for further speedups.

---

## Next Version (In Progress)

* Implement **warp-level reductions** (`__shfl_xor_sync`, etc.)
* **Fuse softmax passes** (exponentiation + reduction + normalization)
* Optimize **memory coalescing** and **occupancy**

---

## Structure

```
cuda-kernels/
├── softmax_v1(naive)/
├── softmax_v2(reduction based)/
├── softmax_v3(warp level optimised)/
└── README.md
```