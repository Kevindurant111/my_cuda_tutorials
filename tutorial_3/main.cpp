#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FindClosestCPU.h"

extern "C" cudaError_t FindClosestGPU(float3* points, int* indices, int count);

using namespace std;

int main() {
    const int count = 10000;
    int *indices = new int[count];
    float3 *points = new float[count];
    for(int i = 0; i < count; i++) {
        points[i].x = (float)((rand()%10000) - 5000);
        points[i].y = (float)((rand()%10000) - 5000);
        points[i].z = (float)((rand()%10000) - 5000);
    }


    int t = 20;
    long averageTime = 0;
    for(int i = 0; i < t; i++) {
        long startTime = clock();
        FindClosestCPU(points, indices, count);
        long finishTime = clock();
        averageTime += finishTime - startTime;
    }
    averageTime /= t;
    cout << "CPU time: " << averageTime << endl;

    delete[] indices;
    delete[] points;
    return 0;
}