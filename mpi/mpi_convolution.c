#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include "mpi.h"

typedef enum {GREY, RGB} color_t;

void Convolution(uint8_t *, uint8_t *, int, int, int, int, int, int, float**, color_t);
void ConvolutionforGREY(uint8_t *, uint8_t *, int, int, int, int, float **);
void ConvolutionforRGB(uint8_t *, uint8_t *, int, int, int, int, float **);
void Usage(int, char **, char **, int *, int *, int *, color_t *);
uint8_t *offset(uint8_t *, int, int, int);
int RowsDivision(int, int, int);

int main(int argc, char** argv) {
    int img_width, img_height, repetitions;
    char* my_img;
    color_t img_type;
    // MPI tasks ahead
    int process_id, n_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    MPI_Status myStatus;

    MPI_Datatype GreyColumn;
    MPI_Datatype RGBColumn;
    MPI_Datatype GreyRow;
    MPI_Datatype RGBRow;
    MPI_Request ToNorth;
    MPI_Request ToSouth;
    MPI_Request ToWest;
    MPI_Request ToEast;
    MPI_Request FromNorth;
    MPI_Request FromSouth;
    MPI_Request FromWest;
    MPI_Request FromEast;

    int North = -1;
    int South = -1;
    int West = -1;
    int East = -1;

    int rows_divided, columns_divided;

    // Checking validity of inputs
    if(process_id == 0) {
        Usage(argc, argv, &my_img, &img_width, &img_height, &repetitions, &img_type);
        // Each process will handle different bits of data
        rows_divided = RowsDivision(n_processes, img_height, img_width);
        if (rows_divided <= 0 || img_height % rows_divided || n_processes % rows_divided || img_width % (columns_divided = n_processes / rows_divided)) {
                fprintf(stderr, "%s: Cannot divide to processes\n", argv[0]);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                return EXIT_FAILURE;
        }
    }
    if (process_id != 0) {
        my_img = malloc((strlen(argv[1])+1) * sizeof(char));
        strcpy(my_img, argv[1]);
    }

    MPI_Bcast(&img_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img_height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&repetitions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rows_divided, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columns_divided, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int RowsPerProcess = img_height / rows_divided;
    int ColumnsPerProcess = img_width / columns_divided;

    MPI_Type_vector(RowsPerProcess, 1, ColumnsPerProcess+2, MPI_BYTE, &GreyColumn);
    MPI_Type_commit(&GreyColumn);
    MPI_Type_vector(RowsPerProcess, 3, 3*ColumnsPerProcess+6, MPI_BYTE, &RGBColumn);
    MPI_Type_commit(&RGBColumn);
    
    MPI_Type_contiguous(ColumnsPerProcess, MPI_BYTE, &GreyRow);
    MPI_Type_commit(&GreyRow);
    MPI_Type_contiguous(3*ColumnsPerProcess, MPI_BYTE, &RGBRow);
    MPI_Type_commit(&RGBRow);

    int starting_row = (process_id / columns_divided) * RowsPerProcess;
    int starting_column = (process_id % columns_divided) * ColumnsPerProcess;

    // Lets create some filters
    int i, j;
    int box_blur[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    int gaussian_blur[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    int edge_detection[3][3] = {{1, 4, 1}, {4, 8, 4}, {1, 4, 1}};
    float **myFilter = malloc(3 * sizeof(float *));
    for (i = 0 ; i < 3 ; i++)
        myFilter[i] = malloc(3 * sizeof(float));
    for (i = 0 ; i < 3 ; i++) {
        for (j = 0 ; j < 3 ; j++){
            // myFilter[i][j] = box_blur[i][j] / 9.0;
            myFilter[i][j] = gaussian_blur[i][j] / 16.0;
            // myFilter[i][j] = edge_detection[i][j] / 28.0;
        }
    }

    uint8_t *src = NULL, *dst = NULL, *tmpbuf = NULL, *tmp = NULL;
    MPI_File fh;
    int filesize, bufsize, nbytes;
    if (img_type == GREY) {
        filesize = img_width * img_height;
        bufsize = filesize / n_processes;
        nbytes = bufsize / sizeof(uint8_t);
        src = calloc((RowsPerProcess+2) * (ColumnsPerProcess+2), sizeof(uint8_t));
        dst = calloc((RowsPerProcess+2) * (ColumnsPerProcess+2), sizeof(uint8_t));
    } else if (img_type == RGB) {
        filesize = img_width*3 * img_height;
        bufsize = filesize / n_processes;
        nbytes = bufsize / sizeof(uint8_t);
        src = calloc((RowsPerProcess+2) * (ColumnsPerProcess*3+6), sizeof(uint8_t));
        dst = calloc((RowsPerProcess+2) * (ColumnsPerProcess*3+6), sizeof(uint8_t));
    }
    if (src == NULL || dst == NULL) {
        fprintf(stderr, "%s: Not enough memory\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    MPI_File_open(MPI_COMM_WORLD, my_img, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (img_type == GREY) {
        for (i = 1 ; i <= RowsPerProcess ; i++) {
            MPI_File_seek(fh, (starting_row + i-1) * img_width + starting_column, MPI_SEEK_SET);
            tmpbuf = offset(src, i, 1, ColumnsPerProcess+2);
            MPI_File_read(fh, tmpbuf, ColumnsPerProcess, MPI_BYTE, &myStatus);
        }
    } else if (img_type == RGB) {
        for (i = 1 ; i <= RowsPerProcess ; i++) {
            MPI_File_seek(fh, 3*(starting_row + i-1) * img_width + 3*starting_column, MPI_SEEK_SET);
            tmpbuf = offset(src, i, 3, ColumnsPerProcess*3+6);
            MPI_File_read(fh, tmpbuf, ColumnsPerProcess*3, MPI_BYTE, &myStatus);
        }
    }
    MPI_File_close(&fh);

    if (starting_row != 0)
        North = process_id - columns_divided;
    if (starting_row + RowsPerProcess != img_height)
        South = process_id + columns_divided;
    if (starting_column != 0)
        West = process_id - 1;
    if (starting_column + ColumnsPerProcess != img_width)
        East = process_id + 1;

    MPI_Barrier(MPI_COMM_WORLD);

    // The timer starts here
    double timer = MPI_Wtime();
    int t;
    for (t = 0 ; t < repetitions ; t++) {
        // Send and request borders
        if (img_type == GREY) {
            if (North != -1) {
                MPI_Isend(offset(src, 1, 1, ColumnsPerProcess+2), 1, GreyRow, North, 0, MPI_COMM_WORLD, &ToNorth);
                MPI_Irecv(offset(src, 0, 1, ColumnsPerProcess+2), 1, GreyRow, North, 0, MPI_COMM_WORLD, &FromNorth);
            }
            if (West != -1) {
                MPI_Isend(offset(src, 1, 1, ColumnsPerProcess+2), 1, GreyColumn,  West, 0, MPI_COMM_WORLD, &ToWest);
                MPI_Irecv(offset(src, 1, 0, ColumnsPerProcess+2), 1, GreyColumn,  West, 0, MPI_COMM_WORLD, &FromWest);
            }
            if (South != -1) {
                MPI_Isend(offset(src, RowsPerProcess, 1, ColumnsPerProcess+2), 1, GreyRow, South, 0, MPI_COMM_WORLD, &ToSouth);
                MPI_Irecv(offset(src, RowsPerProcess+1, 1, ColumnsPerProcess+2), 1, GreyRow, South, 0, MPI_COMM_WORLD, &FromSouth);
            }
            if (East != -1) {
                MPI_Isend(offset(src, 1, ColumnsPerProcess, ColumnsPerProcess+2), 1, GreyColumn,  East, 0, MPI_COMM_WORLD, &ToEast);
                MPI_Irecv(offset(src, 1, ColumnsPerProcess+1, ColumnsPerProcess+2), 1, GreyColumn,  East, 0, MPI_COMM_WORLD, &FromEast);
            }
        } else if (img_type == RGB) {
            if (North != -1) {
                MPI_Isend(offset(src, 1, 3, 3*ColumnsPerProcess+6), 1, RGBRow, North, 0, MPI_COMM_WORLD, &ToNorth);
                MPI_Irecv(offset(src, 0, 3, 3*ColumnsPerProcess+6), 1, RGBRow, North, 0, MPI_COMM_WORLD, &FromNorth);
            }
            if (West != -1) {
                MPI_Isend(offset(src, 1, 3, 3*ColumnsPerProcess+6), 1, RGBColumn,  West, 0, MPI_COMM_WORLD, &ToWest);
                MPI_Irecv(offset(src, 1, 0, 3*ColumnsPerProcess+6), 1, RGBColumn,  West, 0, MPI_COMM_WORLD, &FromWest);
            }
            if (South != -1) {
                MPI_Isend(offset(src, RowsPerProcess, 3, 3*ColumnsPerProcess+6), 1, RGBRow, South, 0, MPI_COMM_WORLD, &ToSouth);
                MPI_Irecv(offset(src, RowsPerProcess+1, 3, 3*ColumnsPerProcess+6), 1, RGBRow, South, 0, MPI_COMM_WORLD, &FromSouth);
            }
            if (East != -1) {
                MPI_Isend(offset(src, 1, 3*ColumnsPerProcess, 3*ColumnsPerProcess+6), 1, RGBColumn,  East, 0, MPI_COMM_WORLD, &ToEast);
                MPI_Irecv(offset(src, 1, 3*ColumnsPerProcess+3, 3*ColumnsPerProcess+6), 1, RGBColumn,  East, 0, MPI_COMM_WORLD, &FromEast);
            }
        }

        // Inner Data Convolution
        Convolution(src, dst, 1, RowsPerProcess, 1, ColumnsPerProcess, ColumnsPerProcess, RowsPerProcess, myFilter, img_type);


        // Request and compute
        if (North != -1) {
            MPI_Wait(&FromNorth, &myStatus);
            Convolution(src, dst, 1, 1, 2, ColumnsPerProcess-1, ColumnsPerProcess, RowsPerProcess, myFilter, img_type);
        }
        if (West != -1) {
            MPI_Wait(&FromWest, &myStatus);
            Convolution(src, dst, 2, RowsPerProcess-1, 1, 1, ColumnsPerProcess, RowsPerProcess, myFilter, img_type);
        }
        if (South != -1) {
            MPI_Wait(&FromSouth, &myStatus);
            Convolution(src, dst, RowsPerProcess, RowsPerProcess, 2, ColumnsPerProcess-1, ColumnsPerProcess, RowsPerProcess, myFilter, img_type);
        }
        if (East != -1) {
            MPI_Wait(&FromEast, &myStatus);
            Convolution(src, dst, 2, RowsPerProcess-1, ColumnsPerProcess, ColumnsPerProcess, ColumnsPerProcess, RowsPerProcess, myFilter, img_type);
        }

        // Corner data
        if (North != -1 && West != -1)
            Convolution(src, dst, 1, 1, 1, 1, ColumnsPerProcess, RowsPerProcess, myFilter, img_type);
        if (West != -1 && South != -1)
            Convolution(src, dst, RowsPerProcess, RowsPerProcess, 1, 1, ColumnsPerProcess, RowsPerProcess, myFilter, img_type);
        if (South != -1 && East != -1)
            Convolution(src, dst, RowsPerProcess, RowsPerProcess, ColumnsPerProcess, ColumnsPerProcess, ColumnsPerProcess, RowsPerProcess, myFilter, img_type);
        if (East != -1 && North != -1)
            Convolution(src, dst, 1, 1, ColumnsPerProcess, ColumnsPerProcess, ColumnsPerProcess, RowsPerProcess, myFilter, img_type);

        // Wait to have sent all borders
        if (North != -1)
            MPI_Wait(&ToNorth, &myStatus);
        if (West != -1)
            MPI_Wait(&ToWest, &myStatus);
        if (South != -1)
            MPI_Wait(&ToSouth, &myStatus);
        if (East != -1)
            MPI_Wait(&ToEast, &myStatus);

        // swap arrays
        tmp = src;
        src = dst;
        dst = tmp;
    }

    timer = MPI_Wtime() - timer;

    char *img_result = malloc((strlen(my_img) + 9) * sizeof(char));
    strcpy(img_result, "blur_");
    strcat(img_result, my_img);
    MPI_File outFile;
    MPI_File_open(MPI_COMM_WORLD, img_result, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outFile);
    if (img_type == GREY) {
        for (i = 1 ; i <= RowsPerProcess ; i++) {
            MPI_File_seek(outFile, (starting_row + i-1) * img_width + starting_column, MPI_SEEK_SET);
            tmpbuf = offset(src, i, 1, ColumnsPerProcess+2);
            MPI_File_write(outFile, tmpbuf, ColumnsPerProcess, MPI_BYTE, MPI_STATUS_IGNORE);
        }
    } else if (img_type == RGB) {
        for (i = 1 ; i <= RowsPerProcess ; i++) {
            MPI_File_seek(outFile, 3*(starting_row + i-1) * img_width + 3*starting_column, MPI_SEEK_SET);
            tmpbuf = offset(src, i, 3, ColumnsPerProcess*3+6);
            MPI_File_write(outFile, tmpbuf, ColumnsPerProcess*3, MPI_BYTE, MPI_STATUS_IGNORE);
        }
    }
    MPI_File_close(&outFile);

    //We must compare the time of all the processes and find the maximum
    double remote_time;
    if (process_id != 0)
        MPI_Send(&timer, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    else {
        for (i = 1 ; i != n_processes ; ++i) {
            MPI_Recv(&remote_time, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &myStatus);
            if (remote_time > timer)
                timer = remote_time;
        }
        printf("%f\n", timer);
    }

    free(src);
    free(dst);
    MPI_Type_free(&RGBColumn);
    MPI_Type_free(&RGBRow);
    MPI_Type_free(&GreyColumn);
    MPI_Type_free(&GreyRow);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

void Convolution(uint8_t *src, uint8_t *dst, int start_row, int last_row, int start_column, int last_column, int img_width, int img_height, float** myFilter, color_t img_type) {
    int i, j;
    if (img_type == GREY) {
        for (i = start_row ; i <= last_row ; i++)
            for (j = start_column ; j <= last_column ; j++)
                ConvolutionforGrey(src, dst, i, j, img_width+2, img_height, myFilter);
    } else if (img_type == RGB) {
        for (i = start_row ; i <= last_row ; i++)
            for (j = start_column ; j <= last_column ; j++)
                ConvolutionforRGB(src, dst, i, j*3, img_width*3+6, img_height, myFilter);
    } 
}

void ConvolutionforGrey(uint8_t *src, uint8_t *dst, int x, int y, int img_width, int img_height, float** myFilter) {
    int i, j, k, l;
    float afterFilter = 0;
    for (i = x-1, k = 0 ; i <= x+1 ; i++, k++)
        for (j = y-1, l = 0 ; j <= y+1 ; j++, l++)
            afterFilter += src[img_width * i + j] * myFilter[k][l];
    dst[img_width * x + y] = afterFilter;
}

void ConvolutionforRGB(uint8_t *src, uint8_t *dst, int x, int y, int img_width, int img_height, float** myFilter) {
    int i, j, k, l;
    float afterFilterforRED = 0, afterFilterforGREEN = 0, afterFilterforBLUE = 0;
    for (i = x-1, k = 0 ; i <= x+1 ; i++, k++)
        for (j = y-3, l = 0 ; j <= y+3 ; j+=3, l++){
            afterFilterforRED += src[img_width * i + j]* myFilter[k][l];
            afterFilterforGREEN += src[img_width * i + j+1] * myFilter[k][l];
            afterFilterforBLUE += src[img_width * i + j+2] * myFilter[k][l];
        }
    dst[img_width * x + y] = afterFilterforRED;
    dst[img_width * x + y+1] = afterFilterforGREEN;
    dst[img_width * x + y+2] = afterFilterforBLUE;
}

uint8_t *offset(uint8_t *src_array, int i, int j, int img_width) {
    return &src_array[img_width * i + j];
}

void Usage(int argc, char **argv, char **my_img, int *img_width, int *img_height, int *repetitions, color_t *img_type) {
    if (argc == 6 && !strcmp(argv[5], "grey")) {
        *my_img = malloc((strlen(argv[1])+1) * sizeof(char));
        strcpy(*my_img, argv[1]);	
        *img_width = atoi(argv[2]);
        *img_height = atoi(argv[3]);
        *repetitions = atoi(argv[4]);
        *img_type = GREY;
    } else if (argc == 6 && !strcmp(argv[5], "rgb")) {
        *my_img = malloc((strlen(argv[1])+1) * sizeof(char));
        strcpy(*my_img, argv[1]);	
        *img_width = atoi(argv[2]);
        *img_height = atoi(argv[3]);
        *repetitions = atoi(argv[4]);
        *img_type = RGB;
    } else {
        fprintf(stderr, "\nError Input!\n%s Image_name width height repetitions [rgb/grey].\n\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }
}

int RowsDivision(int n_processes,int rows,int columns) {
    int perimeter, rows_to, columns_to, best = 0;
    int perimeter_min = rows + columns + 1;
    for (rows_to = 1 ; rows_to <= n_processes ; ++rows_to) {
        if (n_processes % rows_to || rows % rows_to) continue;
        columns_to = n_processes / rows_to;
        if (columns % columns_to) continue;
        perimeter = rows / rows_to + columns / columns_to;
        if (perimeter < perimeter_min) {
            perimeter_min = perimeter;
            best = rows_to;
        }
    }
    return best;
}
