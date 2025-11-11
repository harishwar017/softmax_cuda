
# CUDA Softmax — Fused Kernel (FlashAttention-Inspired online softmax)

### Single-Pass Online Softmax with Shared-Memory Reduction

This version implements a **fused softmax kernel** inspired by the **FlashAttention algorithm** — using an **online softmax formulation** that maintains **numerical stability** and performs all steps (**max**, **exp**, **sum**, and **normalization**) **in a single pass**.

It represents the next stage of evolution from the earlier **naive** and **reduction-based** CUDA kernels.

---

## Overview

| Version             | Description                                                          |
| :------------------ | :------------------------------------------------------------------- |
| **Naive**           | Multiple kernels, atomic operations, heavy global memory traffic.    |
| **Reduction-Based** | Shared-memory reduction, fewer atomics, partial improvement.         |
| **Fused (Current)** | Single-pass, FlashAttention-inspired online softmax.                 |
| **(Next)**          | Warp-level version with shuffle-based reduction for max performance. |

---

## Highlights of the Fused Version

*  **FlashAttention-inspired design:** Uses **online softmax** — dynamically updates max and sum values during streaming, ensuring **numerical stability** without storing intermediate exponentials.
*  **Single-pass kernel:** Combines exponentiation, max reduction, and normalization in one fused operation.
*  **Shared memory reduction:** Efficient intra-block reductions for both max and sum.
*  **Compute-bound performance:** High SM utilization with minimal global memory use.

---

## Profiling Summary

| Metric                      |      Value | Comment                                |
| :-------------------------- | ---------: | :------------------------------------- |
| **Compute (SM) Throughput** | **76.64%** | Strong utilization — compute-bound.    |
| **Memory Throughput**       | **22.40%** | Expected due to reduced global access. |
| **Duration**                | **329 ms** | For 10M elements.                      |
| **DRAM Throughput**         |  **0.57%** | Minimal global memory activity.        |

**Interpretation:**

> The fused kernel achieves **high compute efficiency** while remaining **memory-light**, indicating effective fusion and excellent kernel-level optimization.

---

## 🔍 Key Takeaways

**Single kernel → minimal memory traffic**
**Numerically stable online softmax** (no overflow)
**~77% SM utilization** → compute-bound, efficient
**FlashAttention-inspired design** enables scalability

---

## Current Limitations

* Reduction is still **block-level only** (no warp intrinsics).
* **Cross-block normalization** requires partial synchronization.
* **Memory coalescing** can be further improved.

---

## Next Steps — Warp-Level Optimization

The next iteration will focus on:

* **Warp-level reductions** using `__shfl_xor_sync()` for intra-warp communication.
* **Fully fused multi-block softmax** (inter-warp normalization).
* Improved **memory coalescing** and **occupancy**.
* Optional **half-precision (FP16)** path for bandwidth optimization.

---
