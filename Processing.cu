//
// Created by kevin on 26/09/21.
//

#include "Processing.h"
#include "CUDA_check.h"
#include "Parameters.h"

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define BLOCK_SIZE 32

#if !defined CONSTANT_MEMORY && !defined SHARED_MEMORY
__global__ void
kernel(unsigned char *input, const unsigned long long int *__restrict__ krn, unsigned char *output, int height, int width, int channels,
       int size, double weight) {
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (iy < height && ix < width) {
        int kCenter = size / 2;
        int dx, dy, px, py;

        for (int ic = 0; ic < channels; ic++) {
            // vars "i?" identify image's element
            unsigned long long int newVal = 0;
            for (int ky = 0; ky < size; ky++) {
                for (int kx = 0; kx < size; kx++) {
                    // vars "k?" identify kernel's element
                    dx = kx - kCenter;
                    dy = ky - kCenter;
                    // vars "d?" identify kernel's element's position with respect to the center
                    px = ix + dx;
                    py = iy + dy;
                    // vars "p?" identify the pixel to combine with kernel's element

                    if (px < 0 || px >= width) {      // edge handling: extend
                        px = (px < 0) ? 0 : (width - 1);
                    }
                    if (py < 0 || py >= height) {
                        py = (py < 0) ? 0 : (height - 1);
                    }

                    newVal += (unsigned long long int) input[py * width * channels + px * channels + ic] *
                              krn[ky * size + kx];
                }
            }
            newVal = (unsigned long long int) ((long double) newVal * weight);
            output[iy * width * channels + ix * channels + ic] = (unsigned char) newVal;
        }
    }
}

Image *process(Image *img, Kernel *krn) {
    Image *res = Image_new_empty(img->width, img->height, img->channels);

    unsigned char *d_input;
    unsigned long long int *d_krn;
    unsigned char *d_output;

    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_input, sizeof(unsigned char) * img->width * img->height * img->channels));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_krn, sizeof(unsigned long long int) * krn->size * krn->size));
    CUDA_CHECK_RETURN(
            cudaMalloc((void **) &d_output, sizeof(unsigned char) * img->width * img->height * img->channels));

    CUDA_CHECK_RETURN(cudaMemcpy((void *) d_input, (void *) img->data,
                                 sizeof(unsigned char) * img->width * img->height * img->channels,
                                 cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy((void *) d_krn, (void *) krn->coefficients,
                                 sizeof(unsigned long long int) * krn->size * krn->size,
                                 cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(ceil(((float) img->width) / BLOCK_SIZE), ceil(((float) img->height) / BLOCK_SIZE));

    kernel<<<gridDim, blockDim>>>(d_input, d_krn, d_output, img->height, img->width, img->channels, krn->size,
                                  krn->weight);
    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy((void *) res->data, (void *) d_output,
                                 sizeof(unsigned char) * img->width * img->height * img->channels,
                                 cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_krn);
    cudaFree(d_output);

    return res;
}
#endif

#if defined CONSTANT_MEMORY && !defined SHARED_MEMORY
__constant__ unsigned long long int KERNEL[25 * 25];

__global__ void
kernelConstant(unsigned char *input, unsigned char *output, int height, int width, int channels,
               int size, double weight) {
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (iy < height && ix < width) {
        int kCenter = size / 2;
        int dx, dy, px, py;

        for (int ic = 0; ic < channels; ic++) {
            // vars "i?" identify image's element
            unsigned long long int newVal = 0;
            for (int ky = 0; ky < size; ky++) {
                for (int kx = 0; kx < size; kx++) {
                    // vars "k?" identify kernel's element
                    dx = kx - kCenter;
                    dy = ky - kCenter;
                    // vars "d?" identify kernel's element's position with respect to the center
                    px = ix + dx;
                    py = iy + dy;
                    // vars "p?" identify the pixel to combine with kernel's element

                    if (px < 0 || px >= width) {      // edge handling: extend
                        px = (px < 0) ? 0 : (width - 1);
                    }
                    if (py < 0 || py >= height) {
                        py = (py < 0) ? 0 : (height - 1);
                    }

                    newVal += (unsigned long long int) input[py * width * channels + px * channels + ic] *
                              KERNEL[ky * size + kx];
                }
            }
            newVal = (unsigned long long int) ((long double) newVal * weight);
            output[iy * width * channels + ix * channels + ic] = (unsigned char) newVal;
        }
    }
}

Image *process(Image *img, Kernel *krn) {
    Image *res = Image_new_empty(img->width, img->height, img->channels);

    unsigned char *d_input;
    unsigned char *d_output;

    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_input, sizeof(unsigned char) * img->width * img->height * img->channels));
    CUDA_CHECK_RETURN(
            cudaMalloc((void **) &d_output, sizeof(unsigned char) * img->width * img->height * img->channels));

    CUDA_CHECK_RETURN(cudaMemcpy((void *) d_input, (void *) img->data,
                                 sizeof(unsigned char) * img->width * img->height * img->channels,
                                 cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(KERNEL, (void *) krn->coefficients,
                                         sizeof(unsigned long long int) * krn->size * krn->size));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(ceil(((float) img->width) / BLOCK_SIZE), ceil(((float) img->height) / BLOCK_SIZE));

    kernelConstant<<<gridDim, blockDim>>>(d_input, d_output, img->height, img->width, img->channels, krn->size,
                                          krn->weight);
    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy((void *) res->data, (void *) d_output,
                                 sizeof(unsigned char) * img->width * img->height * img->channels,
                                 cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);

    return res;
}
#endif

#if !defined CONSTANT_MEMORY && defined SHARED_MEMORY
#define TILE_WIDTH BLOCK_SIZE

__global__ void
kernel(unsigned char *input, const unsigned long long int *__restrict__ krn, unsigned char *output, int height,
       int width, int channels,
       int size, double weight, int w) {
    // shared memory
    extern __shared__ unsigned char input_d[];

    // pixel to process
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    // utility values
    int kRadius = size / 2;
    int batches = (w * w) / (TILE_WIDTH * TILE_WIDTH);

    // vars instantiation
    int dest, destY, destX, srcY, srcX;
    unsigned long long int newVal;

    for (int ic = 0; ic < channels; ic++) {
        // shared memory loading
        for (int b = 0; b < batches; b++) {
            dest = threadIdx.y * TILE_WIDTH + threadIdx.x + b * TILE_WIDTH * TILE_WIDTH;
            destY = dest / w;
            destX = dest % w;
            srcY = blockIdx.y * TILE_WIDTH + destY - kRadius;
            srcX = blockIdx.x * TILE_WIDTH + destX - kRadius;

            if (destY < w) {
                if (srcX < 0 || srcX >= width) {        // edge handling: extend
                    srcX = (srcX < 0) ? 0 : (width - 1);
                }
                if (srcY < 0 || srcY >= height) {
                    srcY = (srcY < 0) ? 0 : (height - 1);
                }

                input_d[destY * w + destX] = input[srcY * width * channels + srcX * channels + ic];
            }
        }

        __syncthreads();

        // convolution
        if (iy < height && ix < width) {
            newVal = 0;
            for (int ky = 0; ky < size; ky++) {
                for (int kx = 0; kx < size; kx++) {
                    newVal += (unsigned long long int) input_d[(threadIdx.y + ky) * w + (threadIdx.x + kx)] *
                              krn[ky * size + kx];
                }
            }

            newVal = (unsigned long long int) ((long double) newVal * weight);
            output[iy * width * channels + ix * channels + ic] = (unsigned char) newVal;
        }
        __syncthreads();
    }
}

Image *process(Image *img, Kernel *krn) {
    Image *res = Image_new_empty(img->width, img->height, img->channels);

    unsigned char *d_input;
    unsigned long long int *d_krn;
    unsigned char *d_output;

    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_input, sizeof(unsigned char) * img->width * img->height * img->channels));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_krn, sizeof(unsigned long long int) * krn->size * krn->size));
    CUDA_CHECK_RETURN(
            cudaMalloc((void **) &d_output, sizeof(unsigned char) * img->width * img->height * img->channels));

    CUDA_CHECK_RETURN(cudaMemcpy((void *) d_input, (void *) img->data,
                                 sizeof(unsigned char) * img->width * img->height * img->channels,
                                 cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy((void *) d_krn, (void *) krn->coefficients,
                                 sizeof(unsigned long long int) * krn->size * krn->size,
                                 cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil(((float) img->width) / TILE_WIDTH), ceil(((float) img->height) / TILE_WIDTH));
    int w = TILE_WIDTH + krn->size - 1;

    kernel<<<gridDim, blockDim, sizeof(unsigned char) * w * w>>>(d_input, d_krn, d_output, img->height, img->width,
                                                                 img->channels, krn->size,
                                                                 krn->weight, w);
    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy((void *) res->data, (void *) d_output,
                                 sizeof(unsigned char) * img->width * img->height * img->channels,
                                 cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_krn);
    cudaFree(d_output);

    return res;
}

#endif

#if defined CONSTANT_MEMORY && defined SHARED_MEMORY
#define TILE_WIDTH BLOCK_SIZE
__constant__ unsigned long long int KERNEL[25 * 25];

__global__ void
kernel(unsigned char *input, unsigned char *output, int height,
       int width, int channels,
       int size, double weight, int w) {
    // shared memory
    extern __shared__ unsigned char input_d[];

    // pixel to process
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    // utility values
    int kRadius = size / 2;
    int batches = (w * w) / (TILE_WIDTH * TILE_WIDTH);

    int dest, destY, destX, srcY, srcX;
    unsigned long long int newVal;

    for (int ic = 0; ic < channels; ic++) {
        // shared memory loading
        for (int b = 0; b < batches; b++) {
            dest = threadIdx.y * TILE_WIDTH + threadIdx.x + b * TILE_WIDTH * TILE_WIDTH;
            destY = dest / w;
            destX = dest % w;
            srcY = blockIdx.y * TILE_WIDTH + destY - kRadius;
            srcX = blockIdx.x * TILE_WIDTH + destX - kRadius;

            if (destY < w) {
                if (srcX < 0 || srcX >= width) {        // edge handling: extend
                    srcX = (srcX < 0) ? 0 : (width - 1);
                }
                if (srcY < 0 || srcY >= height) {
                    srcY = (srcY < 0) ? 0 : (height - 1);
                }

                input_d[destY * w + destX] = input[srcY * width * channels + srcX * channels + ic];
            }
        }

        __syncthreads();

        // convolution
        if (iy < height && ix < width) {
            newVal = 0;
            for (int ky = 0; ky < size; ky++) {
                for (int kx = 0; kx < size; kx++) {
                    newVal += (unsigned long long int) input_d[(threadIdx.y + ky) * w + (threadIdx.x + kx)] *
                              KERNEL[ky * size + kx];
                }
            }

            newVal = (unsigned long long int) ((long double) newVal * weight);
            output[iy * width * channels + ix * channels + ic] = (unsigned char) newVal;
        }
        __syncthreads();
    }
}

Image *process(Image *img, Kernel *krn) {
    Image *res = Image_new_empty(img->width, img->height, img->channels);

    unsigned char *d_input;
    unsigned char *d_output;

    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_input, sizeof(unsigned char) * img->width * img->height * img->channels));
    CUDA_CHECK_RETURN(
            cudaMalloc((void **) &d_output, sizeof(unsigned char) * img->width * img->height * img->channels));

    CUDA_CHECK_RETURN(cudaMemcpy((void *) d_input, (void *) img->data,
                                 sizeof(unsigned char) * img->width * img->height * img->channels,
                                 cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(KERNEL, (void *) krn->coefficients,
                                         sizeof(unsigned long long int) * krn->size * krn->size));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil(((float) img->width) / TILE_WIDTH), ceil(((float) img->height) / TILE_WIDTH));
    int w = TILE_WIDTH + krn->size - 1;

    kernel<<<gridDim, blockDim, sizeof(unsigned char) * w * w>>>(d_input, d_output, img->height, img->width,
                                                                 img->channels, krn->size,
                                                                 krn->weight, w);
    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy((void *) res->data, (void *) d_output,
                                 sizeof(unsigned char) * img->width * img->height * img->channels,
                                 cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);

    return res;
}

#endif