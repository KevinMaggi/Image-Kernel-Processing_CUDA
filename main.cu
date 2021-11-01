#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <sys/time.h>
#include "Image.h"
#include "Utils.h"
#include "Kernel.h"
#include "Processing.h"

/**
 * Min value of kernel dimension to test (MUST be odd)
 */
const int KERNEL_DIM_MIN = 3;
/**
 * Max value of kernel dimension to test (MUST be odd)
 */
const int KERNEL_DIM_MAX = 3;
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
const int REPETITIONS = 1;

void saveTextFile(int *kDim, double *times, char *filename) {
    FILE *file = fopen(filename, "w");

    for (int k = 0; k <= (KERNEL_DIM_MAX - KERNEL_DIM_MIN); k += KERNEL_DIM_STEP) {
        fprintf(file, "%d %f\n", kDim[k], times[k]);
    }

    fclose(file);
}

int main() {
    int *kDim = (int *) malloc(sizeof(int) * (KERNEL_DIM_MAX - KERNEL_DIM_MIN + 1));
    double *times = (double *) malloc(sizeof(double) * (KERNEL_DIM_MAX - KERNEL_DIM_MIN + 1));

    for (int k = KERNEL_DIM_MIN; k <= KERNEL_DIM_MAX; k += KERNEL_DIM_STEP) {
        long cumulative_ms = 0;
        Kernel *krn = Kernel_gaussianBlur(k);
        for (int imageIndex = 1; imageIndex <= IMAGE_QUANTITY; imageIndex++) {
            char *inFilename = (char *) malloc(sizeof(char) * 100);
            sprintf(inFilename, "/home/kevin/CLionProjects/Image_Kernel_Processing/Image/Input/%s-%d.jpg",
                    IMAGE_DIMENSION, imageIndex);
            Image *img = loadJPEG(inFilename);
            Image *res = NULL;
            for (int i = 0; i < REPETITIONS; i++) {
                Image_delete(res); // to free memory of previous result and avoid error

                struct timeval start, end;

                gettimeofday(&start, NULL);
                res = process(img, krn);
                gettimeofday(&end, NULL);

                long duration_ms = ((long) end.tv_sec * 1000 + (long) end.tv_usec / 1000) -
                                   ((long) start.tv_sec * 1000 + (long) start.tv_usec / 1000);

                cumulative_ms += duration_ms;
                printf("For kernel dimension = %d on image %d, computation time is %f seconds\n", k, imageIndex,
                       (double) duration_ms / 1000);
            }

            char *outFilename = (char *) malloc(sizeof(char) * 100);
            sprintf(outFilename, "/home/kevin/CLionProjects/Image_Kernel_Processing/Image/Output/%s-%d-processed%d.png",
                    IMAGE_DIMENSION, imageIndex, k);
            savePNG(outFilename, res);

            Image_delete(img);
            Image_delete(res);
            free(inFilename);
            free(outFilename);
        }

        long meanTime = cumulative_ms / (REPETITIONS * IMAGE_QUANTITY);
        kDim[k - KERNEL_DIM_MIN] = k;
        times[k - KERNEL_DIM_MIN] = (double) meanTime / 1000;
        printf("For kernel dimension = %d, MEAN computation time is %f seconds\n", k, (double) meanTime / 1000);

        Kernel_delete(krn);
    }

    char *filename = (char *) malloc(sizeof(char) * 100);
    sprintf(filename, "/home/kevin/CLionProjects/Image_Kernel_Processing/Image/Output/Times/%sExecTimes.txt",
            IMAGE_DIMENSION);
    saveTextFile(kDim, times, filename);

    free(kDim);
    free(times);
    free(filename);

    return 0;
}