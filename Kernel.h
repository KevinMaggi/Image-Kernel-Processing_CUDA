//
// Created by kevin on 19/09/21.
//

#ifndef IMAGE_KERNEL_PROCESSING_KERNEL_H
#define IMAGE_KERNEL_PROCESSING_KERNEL_H

/**
 * Represents a Kernel matrix
 */
typedef struct {
    /**
     * radius of the kernel
     */
    int size;
    /**
     * scalar for normalization
     */
    double weight;
    /**
     * kernel's coefficients
     */
    unsigned long long int *coefficients;
} Kernel;

/**
 * unnormalized weight
 */
#define unnormalized 1

#define Kernel_getSize(krn) ((krn)->size)
#define Kernel_getWeight(krn) ((krn)->weight)
#define Kernel_getCoefficients(krn) ((krn)->coefficients)

#define Kernel_setSize(krn, val) (Kernel_getSize(krn) = (val))
#define Kernel_setWeight(krn, val) (Kernel_getWeight(krn) = (val))
#define Kernel_setCoefficients(krn, val) (Kernel_getCoefficients(krn) = (val))

/**
 * Creates a new Kernel with a given coefficients and metadata
 * @param size kernel's radius
 * @param weight scalar weight
 * @param coefficients matrix values
 * @return Kernel
 */
Kernel *Kernel_new(int size, double weight, unsigned long long int *coefficients);

/**
 * Creates a new Kernel with only metadata
 * @param size kernel's radius
 * @param weight scalar weight
 * @return Kernel
 */
Kernel *Kernel_newEmpty(int size, double weight);

/**
 * Returns a kernel's value
 * @param krn Kernel
 * @param x x index
 * @param y y index
 * @return coefficient
 */
unsigned long long int Kernel_getCoefficient(Kernel *krn, int x, int y);

/**
 * Sets a kernel's value
 * @param krn Kernel
 * @param x x index
 * @param y y index
 * @param coefficient value
 */
void Kernel_setCoefficient(Kernel *krn, int x, int y, unsigned long long int coefficient);

/**
 * Frees the mem allocation of a Kernel
 * @param krn Kernel
 */
void Kernel_delete(Kernel *krn);

/**
 * Creates a pre-built Kernel for box-blur effect
 * @param size radius
 * @return Kernel
 */
Kernel *Kernel_boxBlur(int size);

/**
 * FOR TEST PURPOSE ONLY
 * Creates a pre-built Kernel for identity
 * @param size radius
 * @return Kernel
 */
Kernel *Kernel_identity(int size);

/**
 * Creates a pre-built Kernel for approximate gaussian-blur effect
 * @param size radius
 * @return Kernel
 */
Kernel *Kernel_gaussianBlur(int size);

/**
 * Normalizes the weight
 * @param krn
 */
void Kernel_normalize(Kernel *krn);

#endif //IMAGE_KERNEL_PROCESSING_KERNEL_H
