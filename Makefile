NVCC=nvcc
CUDAFLAGS=
OPT= -g -G

all: main

main: main.o
	${NVCC} ${OPT} -o main main.o

main.o: main.cu
	$(NVCC) ${OPT} $(CUDAFLAGS) -std=c++11 -c main.cu

clean:
	rm *.o main

prof: main.o
	nvprof ./main