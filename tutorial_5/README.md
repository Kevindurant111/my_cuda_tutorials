# Matrix multiplication in CUDA  
This section demonstrates how to use CUDA to accelerate matrix multiplication. We show both the baseline version and the version using tiling technique.  

# Table of Contents

- [Baseline](#Baseline)
- [Usage](#Usage)
- [Disclaimer](#Disclaimer)  

## Baseline  
We all know that matrix multiplication is essentially the sum of element-wise products between the corresponding rows and columns. For example, if A * B = C, then the element in the 1st row and 2nd column of C comes from the sum of element-wise products between the 1st row of A and the 2nd column of B. So, a very simple design idea emerged: we only need to assign a thread to each element of matrix C, and let it calculate the corresponding sum of element-wise products. The baseline code is as follows. Note that in this example, we assume that both A and B are square matrices of the same size, and the thread block size exactly matches the size of matrix C (i.e., the number of rows and columns of C can be evenly divided by blockDim.y and blockDim.x).  
```
__global__ void matrixMul(const int *a, const int *b, int *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  c[row * N + col] = 0;
  for (int k = 0; k < N; k++) {
    c[row * N + col] += a[row * N + k] * b[k * N + col];
  }
}
```

## Usage  
```bash
nvcc -ccbin gcc-7 MatrixMul.cu -o MatrixMul -lstdc++
./MatrixMul
```

## Disclaimer  
The resources of this tutorial are from online videos on YouTube [CUDA Crash Course: Cache Tiled Matrix Multiplication](https://www.youtube.com/watch?v=3xfyiWhtvZw&list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU&index=4&t=634s).