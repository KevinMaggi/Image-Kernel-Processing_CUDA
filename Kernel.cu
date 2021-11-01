//
// Created by kevin on 19/09/21.
//

#include <malloc.h>
#include "Kernel.h"

Kernel *Kernel_new(int size, double weight, unsigned long long int *coefficients) {
    Kernel *krn = (Kernel *) malloc(sizeof(Kernel));

    Kernel_setSize(krn, size);
    Kernel_setWeight(krn, weight);
    Kernel_setCoefficients(krn, coefficients);

    return krn;
}

Kernel *Kernel_newEmpty(int size, double weight) {
    unsigned long long int *coefficients = (unsigned long long int *) malloc(
            sizeof(unsigned long long int) * size * size);
    return Kernel_new(size, weight, coefficients);
}

unsigned long long int Kernel_getCoefficient(Kernel *krn, int x, int y) {
    unsigned long long int *coefficients = Kernel_getCoefficients(krn);
    int size = Kernel_getSize(krn);
    return coefficients[y * size + x];
}

void Kernel_setCoefficient(Kernel *krn, int x, int y, unsigned long long int coefficient) {
    unsigned long long int *coefficients = Kernel_getCoefficients(krn);
    int size = Kernel_getSize(krn);

    coefficients[y * size + x] = coefficient;
}

void Kernel_delete(Kernel *krn) {
    if (krn != NULL) {
        if (Kernel_getCoefficients(krn) != NULL) {
            free(Kernel_getCoefficients(krn));
        }
        free(krn);
    }
}

Kernel *Kernel_boxBlur(int size) {
    Kernel *krn = Kernel_newEmpty(size, 1.0 / (double) (size * size));

    for (int i = 0; i < size * size; i++) {
        Kernel_getCoefficients(krn)[i] = 1;
    }

    return krn;
}

Kernel *Kernel_identity(int size) {
    Kernel *krn = Kernel_newEmpty(size, unnormalized);

    Kernel_getCoefficients(krn)[(size * size - 1) / 2] = 1;

    return krn;
}

Kernel *Kernel_gaussianBlur(int size) {
    unsigned long long int coef = 0;
    unsigned long long int sum = 0;
    unsigned long long int *coefficients = (unsigned long long int *) malloc(sizeof(unsigned long long int) * size);
    for (int j = 0; j < size; j++) {
        if (j == 0)
            coef = 1;
        else
            coef = coef * (size - j) / j;
        coefficients[j] = coef;
        sum += coef;
    }

    Kernel *krn = Kernel_newEmpty(size, 1.0 / (double) (sum * sum));

    for (int i = 0; i < size * size; i++) {
        int col = i % size;
        int row = i / size;
        Kernel_getCoefficients(krn)[i] = (coefficients[row] * coefficients[col]);
    }

    free(coefficients);
    return krn;
}

void Kernel_normalize(Kernel *krn) {
    unsigned long long int sum = 0;
    for (int i = 0; i < krn->size * krn->size; i++) {
        sum += krn->coefficients[i];
    }
    krn->weight = 1.0 / (double) sum;
}