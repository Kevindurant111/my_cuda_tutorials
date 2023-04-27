#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
__device__ const int blockSize = 640;

__global__ void FindClosestGPU(float3* points, int* indices, int* count) {
    if(*count <= 1) {
        return;
    }

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < *count) {
        float distToClosest = 3.40282e38f;
        for(int i = 0; i < *count; i++) {
            if(i == idx) {
                continue;
            }
            float dist = sqrt((points[idx].x - points[i].x) * (points[idx].x - points[i].x) +
            (points[idx].y - points[i].y) * (points[idx].y - points[i].y) +
            (points[idx].z - points[i].z) * (points[idx].z - points[i].z));
            if(dist < distToClosest) {
                distToClosest = dist;
                indices[idx] = i;
            }
        }
    }
}

extern "C" cudaError_t FindClosestGPUCuda(float3* points, int* indices, int count) {
    cudaError_t cudaStatus;
    int* dev_count = nullptr;
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
    
    cudaStatus = cudaMemcpy(dev_points, points, count * sizeof(float3), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_count);
        cudaFree(dev_indices);
        cudaFree(dev_points);
    }

    FindClosestGPU<<<(count / blockSize) + 1, blockSize>>>(dev_points, dev_indices, dev_count);
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
}


__global__ void FindClosestGPUWithBlocking(float3* points, int* indices, int* count) {
    __shared__ float3 sharedPoints[blockSize];
    if(*count <= 1) {
        return;
    }

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float3 thisPoint;
    float distToClosest = 3.40282e38f;
    if(idx < *count) {
        thisPoint = points[idx];

        for(int currentBlockOfPoints = 0; currentBlockOfPoints < gridDim.x; currentBlockOfPoints++) {
            if(threadIdx.x + currentBlockOfPoints * blockSize < *count) {
                sharedPoints[threadIdx.x] = points[threadIdx.x + currentBlockOfPoints * blockSize];
                __syncthreads();
            }
            
            for(int i = 0; i < blockSize; i++) {
                if(i + currentBlockOfPoints * blockSize == idx) {
                    continue;
                }
                float dist = sqrt((thisPoint.x - sharedPoints[i].x) * (thisPoint.x - sharedPoints[i].x) +
                (thisPoint.y - sharedPoints[i].y) * (thisPoint.y - sharedPoints[i].y) +
                (thisPoint.z - sharedPoints[i].z) * (thisPoint.z - sharedPoints[i].z));
                if((dist < distToClosest) && (i + currentBlockOfPoints * blockSize < *count)) {
                    distToClosest = dist;
                    indices[idx] = i + currentBlockOfPoints * blockSize;
                }
            }
            __syncthreads();
        }
    }
}

extern "C" cudaError_t FindClosestGPUCudaWithBlocking(float3* points, int* indices, int count) {
    cudaError_t cudaStatus;
    int* dev_count = nullptr;
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
    
    cudaStatus = cudaMemcpy(dev_points, points, count * sizeof(float3), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_count);
        cudaFree(dev_indices);
        cudaFree(dev_points);
    }

    FindClosestGPUWithBlocking<<<(count / blockSize) + 1, blockSize>>>(dev_points, dev_indices, dev_count);
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
}