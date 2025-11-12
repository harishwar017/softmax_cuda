# CUDA Kernel Optimization for Softmax

* GPU used: Nvidia A100 80GB
* Code in Cuda C++
* Profiling tools used: Nsight Systems, Nsight compute
* N = 10^8 (10 million)
* v4 is the final submission from my end for the assignment
* v5 is just an experimental endeavor for me to learn more about flash attn's implementation

* I evaluated the functional correctness by making sure adding all the softmax of terms to see if they are ~ 1.0

* Kernels you would see in all implementations (except v5):
* Norm_kernel: Finds maximum to subtract it with all the terms to prevent overflow
* Addition_kernel: Computes the denominator for the softmax doing sum(e(x - max))
* Softmax_kernel: With the maximum and the summation computed by the orevious kernels we finally calculate e(x - max)/sum(e(x - max))

* I have ordered my implementations version wise, starting with v1, a naive implementation I did as soon I learnt CUDA with lots of atomic operation for adding, then comes v2 where used block wise reduction and one atomic operation per block for adding while for the norm_kernel, block wise maximums are copied back to host where the global max is calculated (I did this because the version of CUDA I use doesnt supprt atomicMax for floating point). Further in v3 I avoid the device to host memcpy bottleneck by implementing hierarchical reduction. And in v4 I use warp level optimisation in reduction when size is less than 32. Now in v5 i tried to fuse all kernels into one using online softmax and failed miserably.

### Naive → Reduction-Based → Warp-Level Optimized

The goal is to progressively improve GPU efficiency through rectifying bottlenecks:

* version 1 : Naive implementations (baseline)
* version 2 : Reduction-based optimizations
* version 3 : Hierarchical reduction-based optimizations
* version 4 : Warp-level optimisation added
* version 5 : Experimental online softmax with fused kernel (still working on it)

---

## Implementations

| Version                          | Description                                                                     |
| :------------------------------- | :------------------------------------------------------------------------------ |
| **Naive**                        | Direct global memory access, atomic operations, 3 kernel launches.       |
| **Reduction-Based**              | Uses shared-memory reductions and reduced atomic operation in norm_kernel to reduce contention and improve throughput.      |
| **Hierarchical Reduction-Based** | Used Hirerarchical reduction on device to compute global maximum on device avoiding copying to host and computing global maximum |
| **Warp-Level**                 | used warp level optimisation in reduction when the block size is < 32 avoiding __syncthreads |
| **Experimental online softmax**  | Fused all three kernels into one, inpired by online softmax in flash attn paper|

---

## Comparison Overview

# Since softmax is a memory bound opertion I tried optimising for memory operation in subsequent versions

## Naive Implementation:

   # High DRAM Throughput.
   # Low L1/Shared Hit Rate.

## Warp-Level/Fused (Low Access):

    # Low DRAM Throughput (the goal).
    # High L1/Shared Hit Rate.


## Key Takeaways

* **Atomic operations** severely limit performance, reduction-based design solves this.
* **Shared memory** drastically improves efficiency by reducing global memory traffic.
* **warp-level primitives** takes the reduction one step ahead.
* **fused-kernels*** would make it even better (still working on it).

__
## To-do 

* **Fuse softmax passes** (exponentiation + reduction + normalization)

---

## Structure

```
cuda-kernels/
├── softmax_v1(naive)/
├── softmax_v2(reduction based)/
├── softmax_v3(Hierarchical reduction based)/
├── softmax_v4(warp level optimised)/
├── softmax_v5(online softmax with fused kernel)/
└── README.md
```
