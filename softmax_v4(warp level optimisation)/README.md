# Softmax V4

# Improved CUDA Kernels (Hierarchical Reduction Based + Warp reduction)

This folder presents an **optimized version** of the naive CUDA implementations of **softmax** kernels.
Compared to the baseline, this version achieves **higher throughput and efficiency**, primarily due to **reduced atomic contention** and **better parallel reduction** strategies and **warp level optimisation**. Warp level optimisation is the only new change compared to the previous version (v4)

---

## Kernel Profiling Summary

### **Norm Kernels (Hierarchical Reduction)**

> Multiple norm kernels execute in sequence, each performing part of the reduction on-device to avoid the device-to-host copy bottleneck.

| Metric                          |   Kernel 1 | Kernel 2 | Kernel 3 |
| :------------------------------ | ---------: | -------: | -------: |
| **Compute (SM) Throughput [%]** |      72.76 |     6.45 |     0.05 |
| **Duration [µs]**               |     106.24 |     4.83 |     4.51 |
| **Memory Throughput [%]**       |      81.91 |     7.22 |     0.35 |
| **Elapsed Cycles [cycle]**      |    112,447 |    4,974 |    4,548 |
| **L1/TEX Cache Throughput [%]** |      83.91 |    14.16 |    16.16 |
| **SM Active Cycles [cycle]**    | 109,667.65 | 2,530.44 |    25.04 |
| **L2 Cache Throughput [%]**     |      20.32 |     2.36 |     0.35 |
| **SM Frequency [GHz]**          |       1.06 |     1.03 |     1.01 |
| **DRAM Throughput [%]**         |      19.61 |     1.73 |     0.01 |
| **DRAM Frequency [GHz]**        |       1.50 |     1.46 |     1.44 |

---

### **Addition Kernel**

| Metric                          |      Value |
| :------------------------------ | ---------: |
| **Compute (SM) Throughput [%]** |      64.11 |
| **Duration [µs]**               |     121.34 |
| **Memory Throughput [%]**       |      72.13 |
| **Elapsed Cycles [cycle]**      |    127,616 |
| **L1/TEX Cache Throughput [%]** |      75.12 |
| **SM Active Cycles [cycle]**    | 122,425.94 |
| **L2 Cache Throughput [%]**     |      17.92 |
| **SM Frequency [GHz]**          |       1.05 |
| **DRAM Throughput [%]**         |      17.24 |
| **DRAM Frequency [GHz]**        |       1.49 |

---

### **Softmax Kernel**

| Metric                          |     Value |
| :------------------------------ | --------: |
| **Compute (SM) Throughput [%]** |     48.55 |
| **Duration [µs]**               |     65.47 |
| **Memory Throughput [%]**       |     52.12 |
| **Elapsed Cycles [cycle]**      |    68,770 |
| **L1/TEX Cache Throughput [%]** |     20.41 |
| **SM Active Cycles [cycle]**    | 65,755.10 |
| **L2 Cache Throughput [%]**     |     62.63 |
| **SM Frequency [GHz]**          |      1.05 |
| **DRAM Throughput [%]**         |     52.12 |
| **DRAM Frequency [GHz]**        |      1.50 |

---
