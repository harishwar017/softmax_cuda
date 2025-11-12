# Softmax V2

# Improved CUDA Kernels (Reduction-Based)

This folder presents an **optimized version** of the naive CUDA implementations of **softmax** kernels.
Compared to the baseline, this version achieves **higher throughput and efficiency**, primarily due to **reduced atomic contention** and **better parallel reduction** strategies.

## Kernels Implemented

### 1. `norm_kernel`

Performs normalization across input elements (e.g., finding max or sum).

**Performance Summary:**

| Metric                  | Value         |
| :---------------------- | :------------ |
| Compute (SM) Throughput | **72.74 %**   |
| Memory Throughput       | **81.89 %**   |
| Duration                | **106.14 µs** |
| L1/TEX Cache Throughput | **83.91 %**   |
| L2 Cache Throughput     | **20.25 %**   |
| DRAM Throughput         | **19.61 %**   |
| SM Frequency            | **1.06 GHz**  |

**Observation:**

> Excellent utilization of both **compute** and **memory bandwidth** (>80%).

---

### 2. `addition_kernel`

Performs element-wise addition across arrays with optimized reduction.

**Performance Summary:**

| Metric                  | Value         |
| :---------------------- | :------------ |
| Compute (SM) Throughput | **64.26 %**   |
| Memory Throughput       | **72.29 %**   |
| Duration                | **121.79 µs** |
| L1/TEX Cache Throughput | **75.13 %**   |
| L2 Cache Throughput     | **17.90 %**   |
| DRAM Throughput         | **17.30 %**   |
| SM Frequency            | **1.04 GHz**  |

**Observation:**

> Significant improvement over the **baseline addition kernel**, which previously achieved only **~0.3% compute** and **1.6% memory throughput**.
> This version utilizes over **60% of compute capacity** and **70% of memory bandwidth**, thanks to **reduced atomic contention** and **improved reduction strategy**.

---

### 3. `softmax_kernel`

Computes the exponential normalization (Softmax) operation.

**Performance Summary:**

| Metric                  | Value        |
| :---------------------- | :----------- |
| Compute (SM) Throughput | **48.24 %**  |
| Memory Throughput       | **51.66 %**  |
| Duration                | **66.05 µs** |
| L1/TEX Cache Throughput | **20.26 %**  |
| L2 Cache Throughput     | **62.65 %**  |
| DRAM Throughput         | **51.66 %**  |
| SM Frequency            | **1.04 GHz** |

**Observation:**

> Moderate performance, compute and memory utilization both near 50%.

---

## Comparison to Baseline

| Kernel            | Previous Compute Throughput | New Compute Throughput |     Improvement     |
| :---------------- | :-------------------------: | :--------------------: | :-----------------: |
| `norm_kernel`     |           72.66 %           |       **72.74 %**      |          —          |
| `addition_kernel` |            0.30 %           |       **64.26 %**      |    **~213× higher** |
| `softmax_kernel`  |           48.31 %           |       **48.24 %**      |          —          |

**Key Takeaway:**

> The **addition kernel** shows **drastic improvement** due to **reduction-based summation** and **less atomic contention**.
> Other kernels remain bounded by global memory access and launch overheads.

---

## Bottlenecks

Despite performance gains, several limitations persist:

1. **Multiple Kernel Launches for Softmax**

   * Current softmax implementation still uses **three separate kernels** (exp → sum → normalize).
   * Each pass reads/writes from global memory, increasing latency.

2. **Lack of Warp-Level Optimizations**

   * Reduction operations are not optimized using **warp shuffle** or **warp-synchronous primitives**, leaving performance untapped.
  
---

## Planned Improvements (Next Version)

1. **Warp-Level Optimizations**

   * Use warp-level reductions to minimize synchronization and atomic usage.

2. **Single-Pass Softmax Kernel**

   * Combine exponentiation, sum reduction, and normalization within one kernel to reduce global memory transfers.

3. **Kernel Fusion**

   * Fuse operations where possible to reduce kernel launch overhead and improve cache locality.

---
