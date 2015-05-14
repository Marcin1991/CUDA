#! /bin/bash

rm blur_filter
rm kernel.o

nvcc -c kernel.cu
nvcc -ccbin g++ kernel.o main.cpp lodepng.cpp -lcuda -lcudart -o blur_filter
