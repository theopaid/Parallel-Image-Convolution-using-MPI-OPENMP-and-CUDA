#include <stdio.h>
#include <stdlib.h>
#include "cuda_convolution.h"
#include "functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Each thread takes care of one element of vector 'src'
__global__ void kernelConvolutionforGrey(uint8_t *src, uint8_t *dst, int img_width, int img_height) {
    int i, j, k, l;
    // Filter initialization
    int my_filter[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    // get position
    size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    // convolute
    if (0 < x && x < img_height-1 && 0 < y && y < img_width-1) {
        float afterFilter = 0;
        for (i = x-1, k = 0 ; i <= x+1 ; i++, k++)
            for (j = y-1, l = 0 ; j <= y+1 ; j++, l++)
                afterFilter += src[img_width * i + j] * my_filter[k][l] / 16.0;
        dst[img_width * x + y] = afterFilter;
    }
}

__global__ void kernelConvolutionforRGB(uint8_t *src, uint8_t *dst, int img_width, int img_height) {
    int i, j, k, l;
    // Filter inialization
    int my_filter[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    // get position
    size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    // convolute
    if (0 < x && x < img_height-1 && 0 < y && y < 3*img_width-3) {
        float afterFilterRED = 0, afterFIlterGREEN = 0, afterFilterBLUE = 0;
        for (i = x-1, k = 0 ; i <= x+1 ; i++, k++) {
            for (j = (y*3)-3, l = 0 ; j <= (y*3)+3 ; j+=3, l++) {
                afterFilterRED += src[(img_width*3) * i + j]* my_filter[k][l] /16.0;
                afterFIlterGREEN += src[(img_width*3) * i + j+1] * my_filter[k][l] /16.0;
                afterFilterBLUE += src[(img_width*3) * i + j+2] * my_filter[k][l] /16.0;
            }
        }
        dst[img_width*3 * x + (y*3)] = afterFilterRED;
        dst[img_width*3 * x + (y*3)+1] = afterFIlterGREEN;
        dst[img_width*3 * x + (y*3)+2] = afterFilterBLUE;
    }
}

extern "C" void GPU_convolution(uint8_t *src, int img_width, int img_height, int repetitions, color_t img_type)
{
    // Vectors
    uint8_t *d_src, *d_dst, *tmp;
    size_t bytes = (img_type == GREY) ? img_height * img_width : img_height * img_width*3;

    // Allocate memory for each vector on GPU
    CUDA_SAFE_CALL( cudaMalloc(&d_src, bytes * sizeof(uint8_t)) );
    CUDA_SAFE_CALL( cudaMalloc(&d_dst, bytes * sizeof(uint8_t)) );
 
    // Copy host vectors to device memory
    CUDA_SAFE_CALL( cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemset(d_dst, 0, bytes) );

    int t;
    const int blockSize = 16;
    // Convolute "repetition" times
    for (t = 0 ; t < repetitions ; t++) {
        
        if (img_type == GREY) {
            // Specify layout of Grid and Blocks
            int gridX = FRACTION_CEILING(img_height, blockSize);
            int gridY = FRACTION_CEILING(img_width, blockSize);
            dim3 block(blockSize, blockSize);
            dim3 grid(gridX, gridY);
            kernelConvolutionforGrey<<<grid, block>>>(d_src, d_dst, img_width, img_height);
        } else if (img_type == RGB) {
            int gridX = FRACTION_CEILING(img_height, blockSize);
            int gridY = FRACTION_CEILING(img_width*3, blockSize);
            dim3 block(blockSize, blockSize);
            dim3 grid(gridX, gridY);
            kernelConvolutionforRGB<<<grid, block>>>(d_src, d_dst, img_width, img_height);
        }

        // swap arrays
        tmp = d_src;
        d_src = d_dst;
        d_dst = tmp;
    }

    CUDA_SAFE_CALL( cudaGetLastError() );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    
    // Copy array back to host
    if (repetitions%2 == 0) {
        CUDA_SAFE_CALL( cudaMemcpy(src, d_src, bytes, cudaMemcpyDeviceToHost) );
    } else {
           CUDA_SAFE_CALL( cudaMemcpy(src, d_dst, bytes, cudaMemcpyDeviceToHost) );
       }

    // Release device memory
    CUDA_SAFE_CALL( cudaFree(d_src) );
    CUDA_SAFE_CALL( cudaFree(d_dst) );
}