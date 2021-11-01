//
// Created by kevin on 26/09/21.
//

#ifndef IMAGE_KERNEL_PROCESSING_PROCESSING_H
#define IMAGE_KERNEL_PROCESSING_PROCESSING_H

#include "Image.h"
#include "Kernel.h"

/**
 * Operates the Kernel Image Process
 * Edge handling: extend
 * @param img Image
 * @param krn Kernel
 * @return Resulting image
 */
Image *process(Image *img, Kernel *krn);

#endif //IMAGE_KERNEL_PROCESSING_PROCESSING_H
