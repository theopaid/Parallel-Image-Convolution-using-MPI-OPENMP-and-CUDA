#ifndef FUNCTIONS_H
#define	FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "functions.h"

#define CUDA_SAFE_CALL(call) {                                    		\
    cudaError err = call;												\
    if( cudaSuccess != err) {                                           \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",   \
                __FILE__, __LINE__, cudaGetErrorString( err) );         \
        exit(EXIT_FAILURE);                                             \
    } }

#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/denominator)

typedef enum {RGB, GREY} color_t;

int write_info(int, uint8_t *, int);
int read_info(int, uint8_t *, int);
void Initialization(int, char **, char **, int *, int *, int *, color_t *);
uint64_t micro_time(void);

#endif