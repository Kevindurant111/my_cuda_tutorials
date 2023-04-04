# The basic concepts in CUDA
## Host and device  
Host refers to the host or CPU, which is the central processing unit of the computer system running the CUDA program. In a CUDA program, the host is mainly responsible for calling the CUDA API to manage the memory on the device and execute kernel functions. Typically, host allocates memory on the host side, transfers the data to the device, executes the kernel functions, and transfers the results back to the host. host can also control the starting and stopping of the kernel, and can monitor and handle runtime errors.  
Device refers to the GPU, which is the computing device on which the CUDA program runs. In CUDA programs, the device is responsible for executing kernel functions and accessing memory on the GPU. memory on the device includes global memory, shared memory, constant memory, texture memory, etc., and can be accessed by all kernel functions. When executing kernel functions, each thread on the device is assigned some task and accesses its own thread-local memory and registers.  
In CUDA programming, it is often necessary to transfer data from memory on the host to memory on the device for use in kernel functions. The data transfer can be done through functions in the CUDA API, including cudaMemcpy(), cudaMemcpyAsync(), etc. It is important to note that the efficiency of data transfer is important because it can become one of the bottlenecks of the program. Therefore, the way and frequency of data transfer needs to be reasonably designed to maximize the parallel computing power of the GPU.  
## CUDA C Language  
The syntax of CUDA C is basically the same as that of the C language, but CUDA C does not support recursion, variable numbers of arguments, and direct access to system memory.
- CUDA C itself does not restrict the use of recursion, but due to the limitations of GPU hardware architecture and execution model, recursive calls may cause performance degradation or even program crashes.  
The GPU execution model is based on threads and thread blocks, and requires special hardware resources such as shared memory to support parallel computing. This execution model differs significantly from that of the CPU, which requires additional hardware and software support to execute recursive calls on the GPU. Therefore, it is generally recommended to use loops or iterations instead of recursive calls to better adapt to the GPU execution model.  
In addition, recursive calls in CUDA C may cause stack overflow problems. The stack space on the GPU is limited, and if the depth of recursive calls is too large, it may cause stack overflow and program crashes. To avoid this, it is usually necessary to manually control the depth of recursive calls or use non-recursive algorithms to achieve the same functionality.  
In conclusion, although CUDA C supports recursive calls, it is advisable to avoid using them in practical programming to better adapt to the GPU hardware architecture and execution model, and to ensure program performance and stability.    
- In CUDA C, functions cannot use variable numbers of arguments, as the language does not support the std::va_list type or the va_start(), va_arg(), and va_end() functions used in C or C++ to handle variable arguments.  
The lack of support for variable arguments is due to the fact that CUDA C is a subset of the C language, and the C language itself does not have a standard way to handle variable arguments. Instead, CUDA C functions must be defined with a fixed number of arguments, and any additional arguments must be passed using an array or a struct.  
To pass a variable number of arguments to a CUDA C function, one approach is to use an array or a struct to pack the arguments into a single data structure, and then pass the data structure as a single argument to the function. Another approach is to use preprocessor macros to generate multiple versions of the function with different numbers of arguments, and then choose the appropriate version at compile time based on the number of arguments.  
Overall, while CUDA C does not support variable numbers of arguments directly, there are workarounds that can be used to achieve similar functionality.  
# CUDA qualifiers  
In CUDA C, there are three types of functions: global functions, device functions, and host functions.  
- A __global__ function, marked with the __global\_\_ qualifier, is executed on the GPU and can be called from the host. It is typically used to implement a kernel function, which is the entry point for parallel execution on the GPU. A global function can access global memory, shared memory, and other GPU resources, and is executed in parallel by multiple threads in a thread block.  
- A __device__ function, marked with the __device\_\_ qualifier, is also executed on the GPU but can only be called from other device functions or kernel functions. It is typically used to implement reusable code that is used by multiple kernel functions, such as mathematical functions or data structure operations. A device function can access global memory, shared memory, and other GPU resources.  
- A __host__ function, marked with the __host\_\_ qualifier(or no qualifier), is executed on the host CPU and can only be called from other host functions. It is typically used to implement code that sets up the GPU computation, transfers data between the host and the device, or performs other host-related operations. A host function can access host memory, but cannot directly access device memory.  
To implement a parallel computation on the GPU, one typically defines a kernel function as a global function and invokes it from the host using special syntax, such as kernel<<<gridSize, blockSize>>>(args). The kernel function then executes on the GPU, and its results can be transferred back to the host using special memory transfer functions.  
In summary, in CUDA C, global functions are executed on the GPU and can be called from the host, device functions are executed on the GPU and can only be called from other device functions or kernel functions, and host functions are executed on the host CPU and can only be called from other host functions.  
## CUDA API functions  
- cudaMalloc is a CUDA C function used to allocate memory on the device (GPU). It takes a size in bytes as input and returns a pointer to the allocated memory block, or NULL if the allocation fails.  
The function prototype is as follows:
    ```bash
    cudaError_t cudaMalloc(void **devPtr, size_t size);
    ```  
    The first argument, devPtr, is a pointer to a pointer that will store the device memory address of the allocated block. The second argument, size, is the size in bytes of the memory block to allocate.  
- cudaFree is a CUDA C function used to free memory that was previously allocated on the device (GPU) using cudaMalloc, cudaMallocManaged, or other memory allocation functions.  
The function prototype is as follows:  
    ```bash
    cudaError_t cudaFree(void *devPtr);
    ```  
    The devPtr argument is a pointer to the device memory block that should be freed. Once the memory is freed, the memory pointer becomes invalid, and any attempt to access it will result in undefined behavior.  
- cudaMemcpy is a CUDA C function used to copy data between the host (CPU) and the device (GPU), or between different regions of device memory. The function is used to transfer data from one memory location to another.  
The function prototype is as follows:  
    ```bash
    cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
    ```
    The dst argument is a pointer to the destination memory block, while src is a pointer to the source memory block. The count argument specifies the number of bytes to be copied. The kind argument specifies the direction of the copy operation and can take one of the following values:  
    - cudaMemcpyHostToDevice: Copy data from host to device.  
    - cudaMemcpyDeviceToHost: Copy data from device to host. 
    - cudaMemcpyDeviceToDevice: Copy data between different regions of device memory.  
    - cudaMemcpyHostToHost: Copy data between different regions of host memory.
## CUDA_ERROR_LAUNCH_TIMEOUT ERROR   
In CUDA, global function is a function executed on the GPU device for parallel computing. If the running time of a global function exceeds 2 seconds, the CUDA driver will automatically terminate the function and throw a CUDA_ERROR_LAUNCH_TIMEOUT error. This is to avoid long-running functions occupying GPU resources and preventing other applications from using the GPU.  
If the running time of a global function exceeds 2 seconds, you can try the following methods to solve it:  
- Optimize the code: minimize the running time of the global function by optimizing the algorithm, reducing data transfer, using shared memory, and other methods.  
- Split the global function into smaller functions: if the running time of the global function is too long, you can split it into smaller functions and call these functions on the host side. This can avoid running a single function on the GPU for a long time.  
- Increase the CUDA runtime timeout: you can solve the CUDA_ERROR_LAUNCH_TIMEOUT error by setting the CUDA runtime timeout. Before calling the global function on the host side, use the cudaDeviceSetLimit function to set the CUDA runtime timeout. For example:
    ```bash
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, <timeout>);
    ```
    Here, __timeout__ is the timeout time in milliseconds. The longer the timeout time, the longer the GPU occupancy time, so it should be adjusted according to the specific situation.  

Note that although you can increase the CUDA runtime timeout to avoid the CUDA_ERROR_LAUNCH_TIMEOUT error, this is not the best solution. The best way is to minimize the running time of the global function by optimizing the code and splitting functions, etc.


