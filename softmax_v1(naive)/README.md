# Softmax V1

# Naive CUDA Kerne implementation for softmax

This folder contains a set of **CUDA kernel implementations** for my naive implementation of **softmax**

## Kernels Implemented

### 1. `addition_kernel`

Performs element-wise vector addition on the GPU.

**Performance Summary:**

| Metric                  | Value        |
| :---------------------- | :----------- |
| Compute (SM) Throughput | **0.30 %**   |
| Memory Throughput       | **1.65 %**   |
| Duration                | **28.99 ms** |
| DRAM Throughput         | **0.07 %**   |
| L1/TEX Cache Throughput | **0.32 %**   |
| L2 Cache Throughput     | **1.65 %**   |

**Observation:**

> This kernel exhibits **low compute throughput** and **low memory bandwidth utilization**.
> Achieved performance is below 60 % of the device’s peak, typically indicating **latency-bound execution**, possibly due to inefficient memory access and lack of parallelism (atomic add).

---

### 2. `norm_kernel`

Computes the normalization factor (finds max).

**Performance Summary:**

| Metric                  | Value         |
| :---------------------- | :------------ |
| Compute (SM) Throughput | **72.66 %**   |
| Memory Throughput       | **81.80 %**   |
| Duration                | **106.14 µs** |
| L1/TEX Cache Throughput | **83.96 %**   |
| L2 Cache Throughput     | **20.21 %**   |
| DRAM Throughput         | **19.59 %**   |

**Observation:**

> This kernel utilizes **>80 %** of the available compute and memory performance, indicating **good parallel efficiency** and **well-balanced workload distribution**.

---

### 3. `softmax_kernel`

Computes the exponential normalization (Softmax) operation.

**Performance Summary:**

| Metric                  | Value        |
| :---------------------- | :----------- |
| Compute (SM) Throughput | **48.31 %**  |
| Memory Throughput       | **51.86 %**  |
| Duration                | **65.82 µs** |
| L1/TEX Cache Throughput | **20.29 %**  |
| L2 Cache Throughput     | **62.89 %**  |
| DRAM Throughput         | **51.86 %**  |

**Observation:**

> The kernel demonstrates **moderate utilization**, with compute and memory throughput both below 60 %.

---

## Kernel Execution Times

| Kernel            | Execution Time |
| :---------------- | ------------------: |
| `norm_kernel`     |          **106.14 µs** |
| `addition_kernel` |          **28.99 ms** |
| `softmax_kernel`  |           **65.82 µs** |

---

## Bottleneck Analysis

The following issues contribute to low GPU utilization:

1. **Atomic Operations (`atomicAdd`)**

   * The second pass uses `atomicAdd`, causing multiple threads to update the same memory location concurrently.
   * This leads to **race conditions**, **serialization**, and **throughput degradation**.

2. **Direct Global Memory Access**

   * All kernels read/write from **global memory** directly.
   * **No shared memory** (on-chip cache) is used to reduce latency or increase data reuse.

3. **Multiple Kernel Launches**

   * Splitting computation across too many kernels adds **launch overheads** and reduces overall GPU efficiency.

4. **Excessive CPU–GPU Transfers**

   * Frequent **host-device data transfers** increase total runtime due to PCIe latency.

---

## Next Steps (Optimization Ideas)

* Introduce **shared memory** for intermediate reductions and cached reads.
* Replace atomic operations with **parallel reduction strategies**.
* Fuse multiple kernels to reduce launch overhead.
* Minimize host–device synchronization and data movement.

---
