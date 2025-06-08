# AMD FP8 GEMM Challenge Solution

## Table of Contents
1. [Summary](#summary)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Results](#results)
5. [Scaling Methods](#scaling-methods)
6. [License](#license)

## Summary

This repository contains my solution to the AMD FP8 GEMM Challenge, organized by the GPU Mode community. The solution is beginner-friendly and concise—just ~100 lines of code—but still managed to achieve rank 5/163 by the end of the competition.

In addition to the competition solution, this repository includes a comparison of FP8 quantized matrix multiplication (GEMM) with PyTorch’s BF16 GEMM, specifically on the AMD MI300X GPU, evaluating:

- **Memory usage**
- **Accuracy**
- **Throughput**

The code explores three FP8 scaling strategies:
- **Global scaling**
- **Block scaling** (128×128)
- **Row-wise block scaling** (1×128)

