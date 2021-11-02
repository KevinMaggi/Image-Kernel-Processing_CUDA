cmake_minimum_required(VERSION 3.20)
project(Image_Kernel_Processing_CUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Image_Kernel_Processing_CUDA main.cu Image.h Image.cu Utils.h Utils.cu stb_image.h stb_image_write.h Kernel.cu Kernel.h Processing.cu Processing.h CUDA_check.h)

set_target_properties(Image_Kernel_Processing_CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_50,code=sm_50")