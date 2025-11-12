# Softmax V4

# Improved CUDA Kernels (Hierarchical Reduction Based)

This folder presents an **optimized version** of the naive CUDA implementations of **softmax** kernels.
Compared to the baseline, this version achieves **higher throughput and efficiency**, primarily due to **reduced atomic contention**, **better parallel reduction** strategies and reduced host to device and device to host copies.

