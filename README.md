# **CUDA Kernel Optimization for Softmax**
This project gave me a real taste of CUDA optimization and I learned a lot trying to optimise it. I think now I can safely say GPUs aren't a black box anymore, now that I got to work with whats under the hood!

I have tried implementing GPU kernels to calculate the softmax of a given set of numbers. I am making this repo so it is easy for me to organise everything and share it!

**Environment & Setup**

* **GPU Used:** Nvidia A100 80GB
* **Language:** CUDA C++
* **Profiling Tools:** Nsight Systems, Nsight Compute (I have added those metrics in each version's readme file)
* **Input Size:** N = 10^7 (10 million)
* **Notes:**

  * **v4** is the final submission version.
  * **v5** is an experimental attempt to explore FlashAttention’s online softmax approach.
* **Validation:** Verified functional correctness by ensuring the sum of all softmax terms ≈ 1.0

---

## **Kernel Overview**

All implementations (except **v5**) include the following kernels:

* **`Norm_kernel`** – Finds the maximum to subtract from all terms (prevents overflow).
* **`Addition_kernel`** – Computes the denominator for softmax using `sum(exp(x - max))`.
* **`Softmax_kernel`** – Computes the final softmax values as `exp(x - max) / sum(exp(x - max))`.

---
I organized my implementations version-wise, starting with v1, a naive CUDA attempt packed with atomic operations. v2 added block-wise reduction with one atomic per block, and v3 removed the host-device bottleneck using hierarchical reduction. v4 brought in warp-level optimization for smaller blocks, and v5 was my experimental try at fusing everything into one kernel with online softmax, inspired by FlashAttention but not quite there yet.

## **Implementation Evolution**

The versions are arranged in order of increasing optimization and complexity:

**Naive → Reduction-Based → Hierarchical → Warp-Level → Fused**

Goal: **Progressively improve GPU efficiency by removing bottlenecks.**

| Version                                | Description                                                                                                                           |
| :------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| **v1 – Naive**                         | Direct global memory access, heavy use of atomic operations, 3 kernel launches.                                                       |
| **v2 – Reduction-Based**               | Introduces shared-memory reductions and reduces atomic contention; global max computed on host due to lack of `atomicMax` for floats. |
| **v3 – Hierarchical Reduction-Based**  | Implements full on-device hierarchical reduction to eliminate host-device copy bottlenecks.                                           |
| **v4 – Warp-Level Optimized**          | Adds warp-level optimizations for reductions when block size < 32, removing the need for `__syncthreads()`.                           |
| **v5 – Experimental (Online Softmax)** | Attempts to fuse all kernels into one using an online softmax approach inspired by FlashAttention (didn’t work as intended).          |


### **Naive Implementation**

* High **DRAM Throughput**
* Low **L1 / Shared Memory Hit Rate**

### **Warp-Level / Fused Implementations**

* Lower **DRAM Throughput** (goal)
* Higher **L1 / Shared Memory Hit Rate**

---

## **Repository Structure**

```
cuda-kernels/
├── softmax_v1/   (naive)
├── softmax_v2/   (reduction-based)
├── softmax_v3/   (hierarchical reduction-based)
├── softmax_v4/   (warp-level optimized)
├── softmax_v5/   (online softmax with fused kernel)
└── README.md
```

---
