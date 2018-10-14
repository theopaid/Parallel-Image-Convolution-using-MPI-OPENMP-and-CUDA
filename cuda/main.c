#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include "cuda_convolution.h"
#include "functions.h"

int main(int argc, char** argv) {
    int img_input, img_width, img_height, repetitions;
    char *my_img;
    color_t img_type;
    
    Initialization(argc, argv, &my_img, &img_width, &img_height, &repetitions, &img_type);

    // Vectors
    uint8_t *src = NULL;

    uint64_t time_elapsed = micro_time();

    // Get information from the image
    if ((img_input = open(my_img, O_RDONLY)) < 0) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    size_t bytes = (img_type == GREY) ? img_height * img_width : img_height * img_width*3;	
    src = (uint8_t *) calloc(bytes, sizeof(uint8_t));
    read_info(img_input, src, bytes);
    close(img_input);

    GPU_convolution(src, img_width, img_height, repetitions, img_type);

    // Lets create the new image
    int fd;
    char *img_output = (char*) malloc((strlen(my_img) + 9) * sizeof(char));
    strcpy(img_output, "blur_");
    strcat(img_output, my_img);
    if ((fd = open(img_output, O_CREAT | O_WRONLY, 0644)) == -1) {
        fprintf(stderr, "cannot open-create %s\n", img_output);
        return EXIT_FAILURE;
    }
    write_info(fd, src, bytes);
    close(fd);
    free(img_output);

    time_elapsed = micro_time() - time_elapsed;
    double million = 1000 * 1000;
    fprintf(stdout, "Execution time: %.3f sec\n", time_elapsed / million);

    free(src);
    return EXIT_SUCCESS;
}