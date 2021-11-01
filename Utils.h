//
// Created by kevin on 18/09/21.
//

#ifndef IMAGE_KERNEL_PROCESSING_UTILS_H
#define IMAGE_KERNEL_PROCESSING_UTILS_H

#include "Image.h"

#include <stdbool.h>

/**
 * Load a JPEG image and return it as Image
 * @param filename path to JPEG
 * @return Image
 */
Image *loadJPEG(const char *filename);

/**
 * Save an array of clusters as PNG image
 * @param clusters reference to array of clusters
 * @param k number of clusters
 * @param filename path to PNG
 * @param width width of image
 * @param height height of image
 * @return TRUE if success, false otherwise
 */
bool savePNG(const char *filename, Image *img);

#endif //IMAGE_KERNEL_PROCESSING_UTILS_H
