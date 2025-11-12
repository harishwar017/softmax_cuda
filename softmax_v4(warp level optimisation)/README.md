# Softmax V6

# Improved CUDA Kernels (Hierarchical Reduction Based + Warp reduction)

This folder presents an **optimized version** of the naive CUDA implementations of **softmax** kernels.
Compared to the baseline, this version achieves **higher throughput and efficiency**, primarily due to **reduced atomic contention** and **better parallel reduction** strategies and warp level optimisation.
