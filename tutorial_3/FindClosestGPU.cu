#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

__global__ void FindClosestGPU(float3* points, int* indices, int* count) {
    if(*count <= 1) {
        return;
    }

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < *count) {
        float distToClosest = 3.40282e38f;
        float dist = sqrt(points[idx].x * points[idx].x +
                        points[idx].y + points[idx].y +
                        points[idx].z + points[idx].z);
        if(dist < distToClosest) {
            distToClosest = dist;
            indices[idx] = idx;
        }
    }
}

extern "C" cudaError_t FindClosestGPUCuda(float3* points, int* indices, int count) {
    cudaError_t cudaStatus;
    int* dev_count;
    int* dev_indices = nullptr;
    float3* dev_points = nullptr;
    cudaStatus = cudaMalloc((void**)&dev_count, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_count);
    }
    cudaStatus = cudaMalloc((void**)&dev_indices, count * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_count);
        cudaFree(dev_indices);
    }
    cudaStatus = cudaMalloc((void**)&dev_points, count * sizeof(float3));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_count);
        cudaFree(dev_indices);
        cudaFree(dev_points);
    }

    cudaStatus = cudaMemcpy(dev_count, &count, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_count);
        cudaFree(dev_indices);
        cudaFree(dev_points);
    }

    cudaStatus = cudaMemcpy(dev_indices, indices, count * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_count);
        cudaFree(dev_indices);
        cudaFree(dev_points);
    }
    
    cudaStatus = cudaMemcpy(dev_points, points, count * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_count);
        cudaFree(dev_indices);
        cudaFree(dev_points);
    }

    FindClosestGPU<<<(count / 1024) + 1, 1024>>>(dev_points, dev_indices, dev_count);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "FindClosestGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_count);
        cudaFree(dev_indices);
        cudaFree(dev_points);
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        cudaFree(dev_count);
        cudaFree(dev_indices);
        cudaFree(dev_points);
    }

    cudaStatus = cudaMemcpy(indices, dev_indices, count * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_count);
        cudaFree(dev_indices);
        cudaFree(dev_points);
    }

    return;
}