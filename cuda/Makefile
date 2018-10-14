cuda_convolution: main.o functions.o cuda_convolution.o
	nvcc -o cuda_convolution main.o functions.o cuda_convolution.o -O2

main.o: main.c 
	gcc -c main.c -O2

functionss.o: functionss.c
	gcc -c functions.c -O2

cuda_convolution.o: cuda_convolution.cu
	nvcc -c cuda_convolution.cu

clean:
	\rm -f *.o cuda_convolution