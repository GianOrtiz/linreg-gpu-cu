NVCC=nvcc
CUDAFLAGS= -arch=sm_30
OPT= -g -G
RM=/bin/rm -f
all: file

main: file.o generate.o
	${NVCC} ${OPT} -o main file.o generate.o

generate.o: file.cuh generate.cpp
    ${NVCC} ${OPT} ${CUDAFLAGS} -std=c++11 -c generate.cpp

file.o: file.cuh file.cu
    $(NVCC) ${OPT} $(CUDAFLAGS) -std=c++11 -c file.cu

file: file.o generate.o
	${NVCC} ${CUDAFLAGS} -o file file.o generate.o

clean:
    ${RM} *.o file