//
// Created by kevin on 03/11/21.
//

#ifndef IMAGE_KERNEL_PROCESSING_CUDA_PARAMETERS_H
#define IMAGE_KERNEL_PROCESSING_CUDA_PARAMETERS_H

/**
 * Enable the use of constant memory in kernel for kernel
 */
//#define CONSTANT_MEMORY

/**
 * Enable the use of shared memory in kernel
 */
#define SHARED_MEMORY

/**
 * Enable the use of pinned memory for image loading
 */
//#define PINNED_MEMORY

/**
 * Min value of kernel dimension to test (MUST be odd)
 */
const int KERNEL_DIM_MIN = 25;
/**
 * Max value of kernel dimension to test (MUST be odd)
 */
const int KERNEL_DIM_MAX = 25;
/**
 * Step on values of kernel dimension (MUST be even)
 */
const int KERNEL_DIM_STEP = 6;
/**
 * Image dimension to test: 4K, 5K, 6K, 7K or 8K
 */
const char IMAGE_DIMENSION[] = "4K";
/**
 * Number of image of each dimension to test (max 3)
 */
const int IMAGE_QUANTITY = 3;
/**
 * Number of times to test each image
 */
const int REPETITIONS = 2;

#endif //IMAGE_KERNEL_PROCESSING_CUDA_PARAMETERS_H
