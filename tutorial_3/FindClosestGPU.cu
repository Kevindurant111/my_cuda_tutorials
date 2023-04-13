#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

__global__ void FindClosestGPU(float3* points, int* indices, int count) {
    if(count <= 1) {
        return;
    }

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < count) {
        float distToClosest = 3.40282e38f;
        float dist = sqrt(point[idx].x * point[idx].x +
                        point[idx].y + point[idx].y
                        point[idx].z + point[idx].z);
        if(dist < distToClosest) {
            distToClosest = dist;
            indices[idx] = i;
        }
    }
}