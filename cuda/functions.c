#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include "functions.h"

void Initialization(int argc, char **argv, char **my_img, int *img_width, int *img_height, int *repetitions, color_t *img_type) {
    if (argc == 6 && !strcmp(argv[5], "grey")) {
        *my_img = (char *)malloc((strlen(argv[1])+1) * sizeof(char));
        strcpy(*my_img, argv[1]);	
        *img_width = atoi(argv[2]);
        *img_height = atoi(argv[3]);
        *repetitions = atoi(argv[4]);
        *img_type = GREY;
    } else if (argc == 6 && !strcmp(argv[5], "rgb")) {
        *my_img = (char *)malloc((strlen(argv[1])+1) * sizeof(char));
        strcpy(*my_img, argv[1]);	
        *img_width = atoi(argv[2]);
        *img_height = atoi(argv[3]);
        *repetitions = atoi(argv[4]);
        *img_type = RGB;
    } else {
        fprintf(stderr, "Error Input!\n%s image_name width height repetitions [rgb/grey].\n", argv[0]);
        exit(EXIT_FAILURE);
    }
}

int write_info(int fd , uint8_t* buff , int size) {
    int n, sent;
    for (sent = 0 ; sent < size ; sent += n)
        if ((n = write(fd, buff + sent, size - sent)) == -1)
            return -1;
    return sent;
}

int read_info(int fd, uint8_t* buff , int size) {
    int n, sent;
    for (sent = 0 ; sent < size ; sent += n)
        if ((n = read(fd, buff + sent, size - sent)) == -1)
            return -1;
    return sent;
}

uint64_t micro_time(void) {
    struct timeval tv;
    assert(gettimeofday(&tv, NULL) == 0);
    return tv.tv_sec * 1000 * 1000 + tv.tv_usec;
}