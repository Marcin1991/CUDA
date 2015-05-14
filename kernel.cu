#include "kernel.h"
#include <iostream>
#include <cmath>

#define FILTER_SIZE 5

/*Kernel blur function -> box blur used defined size matrix*/
__global__ void cudaBlurFilter(int width, int height, unsigned char* in_image, unsigned char* out_image) {

    unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
    
    int x = offset % width;
    int y = (offset-x)/width;
    
    int size = FILTER_SIZE;

    if(offset < width*height) {

        float red = 0;
        float green = 0;
        float blue = 0;
        
        int fields_count = 0;
        
        //main loops, count average values of max = size * size pixels
        for(int ox = -size; ox < size+1; ++ox) {
            for(int oy = -size; oy < size+1; ++oy) {
                
                if((x+ox) > -1 && (x+ox) < width && (y+oy) > -1 && (y+oy) < height) {
                    
                    int currentoffset = (offset+ox+oy*width)*3;
                    
                    out_red += in_image[currentoffset]; 
                    out_green += in_image[currentoffset+1];
                    out_blue += in_image[currentoffset+2];
                    
                    fields_count++;
                }
            }
        }
        
        /*save results to output image array*/
        out_image[offset*3] = red/fields_count;
        out_image[offset*3+1] = green/fields_count;
        out_image[offset*3+2] = blue/fields_count;
        
        }
}


void filter(int width, int height, unsigned char* in_image, unsigned char* out_image) {

    /*malloc two arrays for images on device*/
    unsigned char* dev_input, dev_output;
    cudaMalloc( (void**) &dev_input, width*height*3*sizeof(unsigned char));
    cudaMalloc( (void**) &dev_output, width*height*3*sizeof(unsigned char));

    /*copy data from host to device memory*/
    cudaMemcpy( dev_input, in_image, width*height*3*sizeof(unsigned char), cudaMemcpyHostToDevice );
 
    dim3 blockDims(512,1,1);
    dim3 gridDims((unsigned int) ceil((double)(width*height*3/blockDims.x)), 1, 1 );

    cudaBlurFilter<<<gridDims, blockDims>>>(width, height, dev_input, dev_output); 

    /*copy results to host*/
    cudaMemcpy(out_image, dev_output, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost );

    cudaFree(dev_input);
    cudaFree(dev_output);
}

