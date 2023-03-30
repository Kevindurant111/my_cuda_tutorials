# A simple CUDA-based addition function.  
This tutorial aims to demonstrate how to write a simple CUDA project, including how to write CMakeLists, .cu, and main function files.
## Environment configuration.
Of course, the prerequisite is that you have already installed the CUDA toolkit and your host has at least one NVIDIA graphics card. If you are unsure, you can use 'nvidia-smi' to check the driver version and graphics card status.
![nvidia-smi](./images/nvidia-smi.png)
Also, you can use 'nvcc -V' to check the version of the CUDA toolkit.
![nvcc-V](./images/nvcc-V.png)  
As you can see, my Linux host (running Ubuntu 20.04) has a NVIDIA T400 compute card with 4GB of memory. The version of CUDA toolkit is 10.1. Before using the CUDA toolkit, make sure that the corresponding versions of the compilers (gcc and g++) match. Generally speaking, if your versions of gcc and g++ are too high, you can use the following method to downgrade to gcc-7 and g++-7 as an example:
```bash
# Check the current version of gcc
gcc --version
# Downgrade the versions of g++ and gcc
sudo apt-get update
sudo apt-get install gcc-7 g++-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100
# Check the version of gcc after downgrading
gcc --version
```
## Usage.
```bash
git clone https://github.com/Kevindurant111/my_cuda_tutorials.git
cd my_cuda_tutorials/tutorial_1/
mkdir build
cd build
cmake ..
make
./cuda_example
```
Hopefully, you should see the following output:
![result](./images/result.png)
## Disclaimer.
The code in this article comes from [csdn](https://blog.csdn.net/comedate/article/details/109347874), and has been appropriately modified to ensure that it compiles and runs correctly.