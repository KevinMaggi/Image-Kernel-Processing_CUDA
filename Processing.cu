//
// Created by kevin on 26/09/21.
//

#include "Processing.h"

Image *process(Image *img, Kernel *krn) {
    Image *res = Image_new_empty(img->width, img->height, img->channels);

    unsigned long long int newVal;
    int kCenter = krn->size / 2;
    int dx, dy, px, py;

    for (int iy = 0; iy < img->height; iy++) {
        for (int ix = 0; ix < img->width; ix++) {
            for (int ic = 0; ic < img->channels; ic++) {
                // vars "i?" identify image's element
                newVal = 0;
                for (int ky = 0; ky < krn->size; ky++) {
                    for (int kx = 0; kx < krn->size; kx++) {
                        // vars "k?" identify kernel's element
                        dx = kx - kCenter;
                        dy = ky - kCenter;
                        // vars "d?" identify kernel's element's position with respect to the center
                        px = ix + dx;
                        py = iy + dy;
                        // vars "p?" identify the pixel to combine with kernel's element

                        if (px < 0 || px >= img->width) {      // edge handling: extend
                            px = (px < 0) ? 0 : (img->width - 1);
                        }
                        if (py < 0 || py >= img->height) {
                            py = (py < 0) ? 0 : (img->height - 1);
                        }

                        newVal += (unsigned long long int) Image_getPixel(img, px, py, ic) *
                                  Kernel_getCoefficient(krn, ky, kx);
                    }
                }
                newVal = (unsigned long long int) ((long double) newVal * krn->weight);
                Image_setPixel(res, ix, iy, ic, (unsigned char) newVal);
            }
        }
    }

    return res;
}