//
// Created by kevin on 18/09/21.
//

#include "Image.h"
#include <stdlib.h>
#include "Parameters.h"
#include "CUDA_check.h"

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

Image *Image_new(int width, int height, int channels, unsigned char *data) {
    Image *img = (Image *) malloc(sizeof(Image));

    Image_setWidth(img, width);
    Image_setHeight(img, height);
    Image_setChannels(img, channels);
    Image_setPitch(img, width * channels);

#ifdef PINNED_MEMORY
    unsigned char *dataPinned;
    CUDA_CHECK_RETURN(cudaMallocHost((void**)&dataPinned, sizeof(unsigned char) * width * height * channels));
    memcpy(dataPinned, data, sizeof(unsigned char) * width * height * channels);
    Image_setData(img, dataPinned);
#else
    Image_setData(img, data);
#endif

    return img;
}

Image *Image_new_empty(int width, int height, int channels) {
#ifdef PINNED_MEMORY
    unsigned char *data;
    CUDA_CHECK_RETURN(cudaMallocHost((void**)&data, sizeof(unsigned char) * width * height * channels));
#else
    unsigned char *data = (unsigned char *) malloc(sizeof(unsigned char) * width * height * channels);
#endif

    return Image_new(width, height, channels, data);
}

void Image_delete(Image *img) {
    if (img != NULL) {
        if (Image_getData(img) != NULL) {
#ifdef PINNED_MEMORY
            CUDA_CHECK_RETURN(cudaFreeHost(Image_getData(img)));
#else
            free(Image_getData(img));
#endif
        }
        free(img);
    }
}

void Image_setPixel(Image *img, int x, int y, int c, unsigned char val) {
    unsigned char *data = Image_getData(img);
    int channels = Image_getChannels(img);
    int pitch = Image_getPitch(img);

    data[y * pitch + x * channels + c] = val;
}

unsigned char Image_getPixel(Image *img, int x, int y, int c) {
    unsigned char *data = Image_getData(img);
    int channels = Image_getChannels(img);
    int pitch = Image_getPitch(img);

    return data[y * pitch + x * channels + c];
}
