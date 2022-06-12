NVCC=nvcc
CUDAFLAGS=
OPT= -g -G

all: main

main: main.o generate.o
	${NVCC} ${OPT} -o main main.o generate.o

generate.o: generate.cpp
	${NVCC} ${OPT} ${CUDAFLAGS} -std=c++11 -c generate.cpp

main.o: main.cu
	$(NVCC) ${OPT} $(CUDAFLAGS) -std=c++11 -c main.cu

clean:
	rm *.o main

prof: main.o
	nvprof ./main