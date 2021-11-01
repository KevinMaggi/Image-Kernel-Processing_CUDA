//
// Created by kevin on 18/09/21.
//

#ifndef IMAGE_KERNEL_PROCESSING_IMAGE_H
#define IMAGE_KERNEL_PROCESSING_IMAGE_H


/**
 * Represents an image with data and its metadata
 */
typedef struct {
    int width;
    int height;
    int channels;
    int pitch;
    unsigned char *data;
} Image;

/**
 * RGB channels
 */
#define tri_color 3
/**
 * Gray-scale channel
 */
#define monochrome 1

#define Image_getWidth(img) ((img)->width)
#define Image_getHeight(img) ((img)->height)
#define Image_getChannels(img) ((img)->channels)
#define Image_getPitch(img) ((img)->pitch)
#define Image_getData(img) ((img)->data)

#define Image_setWidth(img, val) (Image_getWidth(img) = (val))
#define Image_setHeight(img, val) (Image_getHeight(img) = (val))
#define Image_setChannels(img, val) (Image_getChannels(img) = (val))
#define Image_setPitch(img, val) (Image_getPitch(img) = (val))
#define Image_setData(img, val) (Image_getData(img) = (val))

/**
 * Creates a new image from data and metadata
 * @param width
 * @param height
 * @param channels
 * @param data
 * @return Image
 */
Image *Image_new(int width, int height, int channels, unsigned char *data);

/**
 * Creates a new empty image with only metadata
 * @param width
 * @param height
 * @param channels
 * @return
 */
Image *Image_new_empty(int width, int height, int channels);

/**
 * Returns the value of a channel of a given pixel
 * @param img Image
 * @param x x-axis coordinate
 * @param y y-axis coordinate
 * @param c channel
 * @return value
 */
unsigned char Image_getPixel(Image *img, int x, int y, int c);

/**
 * Sets the value of a channel of a given pixel
 * @param img Image
 * @param x x-axis coordinate
 * @param y y-axis coordinate
 * @param c channel
 * @param val value
 */
void Image_setPixel(Image *img, int x, int y, int c, unsigned char val);

/**
 * Frees the mem allocation of an image
 * @param img Image
 */
void Image_delete(Image *img);

#endif //IMAGE_KERNEL_PROCESSING_IMAGE_H
