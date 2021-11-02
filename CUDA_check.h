//
// Created by kevin on 02/11/21.
//

#ifndef IMAGE_KERNEL_PROCESSING_CUDA_CUDA_CHECK_H
#define IMAGE_KERNEL_PROCESSING_CUDA_CUDA_CHECK_H

#include <iostream>

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("<< err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

#endif //IMAGE_KERNEL_PROCESSING_CUDA_CUDA_CHECK_H
