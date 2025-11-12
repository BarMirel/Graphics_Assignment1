#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <iostream>

unsigned char clip(int value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return static_cast<unsigned char>(value);
}

void grayscale(const unsigned char* input, unsigned char* output, int width, int height){
    for (int i = 0; i < width * height; ++i) {
        unsigned char r = input[i * 4 + 0];
        unsigned char g = input[i * 4 + 1];
        unsigned char b = input[i * 4 + 2];
        output[i] = clip(static_cast<int>(0.299f*r + 0.587f*g + 0.114f*b));
    }
}

void gaussian_filter(const unsigned char* input, unsigned char* output, int width, int height) {
    int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    int kernelNormalization = 16;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int weightedSum = 0; 
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixelX = x + kx;
                    int pixelY = y + ky;
                    if (pixelX < 0) pixelX = 0;
                    if (pixelX >= width) pixelX = width - 1;
                    if (pixelY < 0) pixelY = 0;
                    if (pixelY >= height) pixelY = height - 1;
                    unsigned char pixelValue = input[pixelY * width + pixelX];
                    int kernelValue = kernel[ky + 1][kx + 1];
                    weightedSum += pixelValue * kernelValue;
                }
            }
            int normalizedSum = weightedSum / kernelNormalization;
            output[y * width + x] = clip(normalizedSum);
        }
    }
}

int main(void)
{
    std::string filepath = "res/textures/Lenna.png";
    int width, height, comps;
    int req_comps = 4;
    unsigned char * buffer = stbi_load(filepath.c_str(), &width, &height, &comps, req_comps);
    
    unsigned char* gray_buffer = new unsigned char[width * height];
    grayscale(buffer, gray_buffer, width, height);
    int result = stbi_write_png("res/textures/Grayscale.png", width, height, 1, gray_buffer, width);
    std::cout << "Grayscale result: " << result << std::endl;

    unsigned char* smooth_buffer = new unsigned char[width * height];
    gaussian_filter(gray_buffer, smooth_buffer, width, height);
    result = stbi_write_png("res/textures/smoothed.png", width, height, 1, smooth_buffer, width);
    std::cout << "Gaussian result: " << result << std::endl;
    delete[] gray_buffer;
    delete[] smooth_buffer;

    stbi_image_free(buffer);
    return 0;
}