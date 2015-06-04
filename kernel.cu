#include "kernel.h"
#include <iostream>
#include <cmath>

#define FILTER_SIZE 5
#define BLOCK_DIMENSION 512

/*Kernel blur function -> box blur used defined size matrix*/
/*It's count average of pixel neighbours*/
/*The same for red value, green value and blue value*/

__global__ void cudaBlurFilter(int width, int height, unsigned char* in_image, unsigned char* out_image) {

    /*offset of current pixel*/
    unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
    
    int x = offset % width;
    int y = (offset-x)/width;
    
    int size = FILTER_SIZE;

    if(offset < width*height) {

        double red = 0;
        double green = 0;
        double blue = 0;
        
        int fields_count = 0;
        
        //main loops, count average values of max = size * size pixels
        for(int ox = -size; ox < size+1; ++ox) {
            for(int oy = -size; oy < size+1; ++oy) {
                
                if((x+ox) > -1 && (x+ox) < width && (y+oy) > -1 && (y+oy) < height) {
                    
                    int currentoffset = (offset+ox+oy*width)*3;
                    
					//update sum of pixel values
                    red += in_image[currentoffset]; 
                    green += in_image[currentoffset+1];
                    blue += in_image[currentoffset+2];
                    
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
    unsigned char* dev_input;
	unsigned char* dev_output;

	int num_devices, device;
	
	int blockNum;
	int threadNum;

	//device selection
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 1) {
		  int max_multiprocessors = 0, max_device = 0;
		  for (device = 0; device < num_devices; device++) {
				  cudaDeviceProp properties;
				  cudaGetDeviceProperties(&properties, device);
				  std::cout<<"device: "<<device<<" with multiprocesors: " << properties.multiProcessorCount << std::endl;
				  if (max_multiprocessors < properties.multiProcessorCount) {
						  max_multiprocessors = properties.multiProcessorCount;
						  max_device = device;
				  }
		  }
		  cudaSetDevice(max_device);
		  std::cout<<"choosed one: "<<max_device << std::endl;
		  
		  std::cout<<"Would you like to change it? [y/n]"<<std::endl;
		  char option;
		  std::cin>>option;
		  int d = max_device;
		  if(option == 'y'){
			std::cout<<"type device number"<<std::endl;
			std::cin>>d;
			std::cout<<"setting device: "<<d<<std::endl;
			cudaSetDevice(d);
		  }
		  
		  cudaDeviceProp properties;
		  cudaGetDeviceProperties(&properties, device);
		  
	}


    cudaMalloc( (void**) &dev_input, width*height*3*sizeof(unsigned char));
    cudaMalloc( (void**) &dev_output, width*height*3*sizeof(unsigned char));

    /*copy data from host to device memory*/
    cudaMemcpy( dev_input, in_image, width*height*3*sizeof(unsigned char), cudaMemcpyHostToDevice);
 
//	dim3 gridDims(16,1,1);
//	dim3 blockDims(512,1,1);

    dim3 blockDims(BLOCK_DIMENSION,1,1);
    dim3 gridDims((unsigned int) ceil((double)(width*height*3/blockDims.x)), 1, 1 );

	std::cout << "calling filter" << std::endl;
	/* kernel invokation */
    cudaBlurFilter<<<gridDims, blockDims>>>(width, height, dev_input, dev_output); 
	/* kernel invokation */

    /*copy results to host*/
    cudaMemcpy(out_image, dev_output, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_output);
}

