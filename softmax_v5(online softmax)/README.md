
# CUDA Softmax — Fused Kernel (FlashAttention-Inspired online softmax)

### Single-Pass Online Softmax with Shared-Memory Reduction

I have implemented one block to do all the compute, further I would implement hierarchical reduction. Since it is 

This version implements a **fused softmax kernel** inspired by the **FlashAttention algorithm**. using an **online softmax formulation** that maintains **numerical stability** and performs all steps (**max**, **exp**, **sum**, and **normalization**) **in a single pass**.

It represents the next stage of evolution from the earlier **naive** and **reduction-based** CUDA kernels.

---

## Profiling Summary

| Metric                      |      Value | Comment                                |
| :-------------------------- | ---------: | :------------------------------------- |
| **Compute (SM) Throughput** | **76.64%** | Strong utilization, compute-bound.    |
| **Memory Throughput**       | **22.40%** | Expected due to reduced global access. |
| **Duration**                | **329 ms** | Too high compared to previous          |
| **DRAM Throughput**         |  **0.57%** | Minimal global memory activity.        |

**Interpretation:**

> The fused kernel achieves **high compute efficiency** but very high latency since one block is doing all the work
---

