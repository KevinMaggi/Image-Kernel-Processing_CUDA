//
// Created by kevin on 18/09/21.
//

#include "Utils.h"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

Image *loadJPEG(const char *filename) {
    Image *img = NULL;
    int width, height, channels;

    unsigned char *rgb_image = stbi_load(filename, &width, &height, &channels, tri_color);

    if (rgb_image != NULL) {
        img = Image_new(width, height, channels, rgb_image);
    }

    return img;
}

bool savePNG(const char *filename, Image *img) {
    return stbi_write_png(filename, img->width, img->height, img->channels, (void *) img->data,
                          img->width * img->channels) ? true : false;
}