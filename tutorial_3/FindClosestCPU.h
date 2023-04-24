#include <cuda.h>
#include <cmath>

void FindClosestCPU(float3 * points, int* indices, int count) {
    if(count <= 1) {
        return;
    }
    for(int curPoint = 0; curPoint < count; curPoint++) {
        float distToClosest = 3.40282e38f;
        for(int i = 0; i < count; i++) {
            if(i == curPoint) {
                continue;
            }
            float dist = sqrt(points[curPoint].x * points[curPoint].x +
            points[curPoint].y + points[curPoint].y +
            points[curPoint].z + points[curPoint].z);
            if(dist < distToClosest) {
                distToClosest = dist;
                indices[curPoint] = i;
            }
        }
    }
}