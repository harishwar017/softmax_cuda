# **CUDA Kernel Optimization for Softmax**

**Environment & Setup**

* **GPU Used:** Nvidia A100 80GB
* **Language:** CUDA C++
* **Profiling Tools:** Nsight Systems, Nsight Compute
* **Input Size:** N = 10⁸ (100 million)
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

## **Implementation Evolution**

The versions are arranged in order of increasing optimization and complexity:

| Version                                | Description                                                                                                                           |
| :------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| **v1 – Naive**                         | Direct global memory access, heavy use of atomic operations, 3 kernel launches.                                                       |
| **v2 – Reduction-Based**               | Introduces shared-memory reductions and reduces atomic contention; global max computed on host due to lack of `atomicMax` for floats. |
| **v3 – Hierarchical Reduction-Based**  | Implements full on-device hierarchical reduction to eliminate host-device copy bottlenecks.                                           |
| **v4 – Warp-Level Optimized**          | Adds warp-level optimizations for reductions when block size < 32, removing the need for `__syncthreads()`.                           |
| **v5 – Experimental (Online Softmax)** | Attempts to fuse all kernels into one using an online softmax approach inspired by FlashAttention (didn’t work as intended).          |

---

### **Progression Summary**

> **Naive → Reduction-Based → Hierarchical → Warp-Level → Fused**

Goal: **Progressively improve GPU efficiency by removing bottlenecks.**

* **v1:** Naive baseline
* **v2:** Shared-memory reduction
* **v3:** Hierarchical reduction (fully on-device)
* **v4:** Warp-level optimization
* **v5:** Experimental fused online softmax

---

## **Comparison Overview**

Since softmax is a **memory-bound operation**, each version focuses on improving memory access patterns.

### **Naive Implementation**

* High **DRAM Throughput**
* Low **L1 / Shared Memory Hit Rate**

### **Warp-Level / Fused Implementations**

* Lower **DRAM Throughput** (goal)
* Higher **L1 / Shared Memory Hit Rate**

---

## **Key Takeaways**

* **Atomic operations** severely limit performance; reduction-based approaches mitigate this.
* **Shared memory** drastically improves throughput by reducing global memory traffic.
* **Warp-level primitives** push performance further by optimizing reductions at the warp level.
* **Fused kernels** (e.g., online softmax) could yield further gains — still in progress.

---

## **To-Do**

* [ ] **Fuse softmax passes** — combine exponentiation, reduction, and normalization into a single efficient kernel.

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

Would you like me to format it in **Markdown for GitHub** style (with collapsible sections or table of contents), or keep it as a **plain-text technical report style** like above?
