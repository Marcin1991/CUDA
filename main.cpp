#include "lodepng.h"
#include "kernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <cstdlib>

#define DEFAULT_PIXEL 255

int main(int argc, char** argv) {
    
    /*Check if run properly*/
    if(argc != 3) {

        std::cout << "Input or output doesn't found!" << std::endl;

        return 0;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    std::vector<unsigned char> in_image; //input image
    unsigned int width, height;

    // use loadpng to load content of .png file
    unsigned error = lodepng::decode(in_image, width, height, input_file);
    
    // convert data from rgba to rgb
    unsigned char* input_image = new unsigned char[(in_image.size()*3)/4];
    unsigned char* output_image = new unsigned char[(in_image.size()*3)/4];
    int pointer = 0;
    for(int i = 0; i < in_image.size(); ++i) {
       if((i+1) % 4 != 0) {
           input_image[pointer] = in_image.at(i);
           output_image[pointer] = DEFAULT_PIXEL;
           pointer++;
       }
    }

    /*invoke filter function from kernel.cu*/
    filter(width, height, input_image, output_image); 

    // Prepare data for output
    std::vector<unsigned char> out_image;
    for(int i = 0; i < in_image.size(); ++i) {
        out_image.push_back(output_image[i]);
        if((i+1) % 3 == 0) {
            out_image.push_back(DEFAULT_PIXEL);
        }
    }
    
    // Output the data
    error = lodepng::encode(output_file, out_image, width, height);

    //if there's an error, display it
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    delete[] input_image;
    delete[] output_image;
    return 0;

}



