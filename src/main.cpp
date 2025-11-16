#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <iostream>
#include <math.h>

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

void halftone(const unsigned char* input, unsigned char* output, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char gray = input[y * width + x];
            int baseX = x * 2;
            int baseY = y * 2;

            unsigned char color00, color01, color10, color11;

            if (gray < 256*0.2) {
                color00 = color01 = color10 = color11 = 0;
            } else if (gray < 256*0.4) {
                color00 = 0; color01 = 0;
                color10 = 255;   color11 = 0;
            } else if (gray < 256*0.6) {
                color00 = 0; color01 = 255;
                color10 = 255;   color11 = 0;
            } else if (gray < 256*0.8) {
                color00 = 0; color01 = 255;
                color10 = 255;   color11 = 255;
            } else {
                color00 = color01 = color10 = color11 = 255;
            }
            output[baseY * width * 2 + baseX] = color00;
            output[baseY * width * 2 + (baseX + 1)] = color01;
            output[(baseY + 1) * width * 2 + baseX] = color10;
            output[(baseY + 1) * width * 2 + (baseX + 1)] = color11;
        }
    }
}

void floyd_steinberg(const unsigned char* input, unsigned char* output, int width, int height){
    float* buffer = new float[width * height]; // New buffer for allowing changes in the array (needed for calculation later)
    for (int i = 0; i < width * height; i++)
        buffer[i] = input[i];

    const float step = 255.0f / 15.0f; //Gives 16 levels of intensity, using / 16 will give an ec=xtra level because of the rounding later

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            float oldPixel = buffer[idx];
            float newPixel = round(oldPixel / step) * step;
            output[idx] = clip((int)newPixel);
            float error = oldPixel - newPixel;
            if (x + 1 < width)
                buffer[idx + 1] += error * 7.0f / 16.0f;
            if (y + 1 < height) {
                buffer[idx + width] += error * 5.0f / 16.0f;
                if (x > 0)
                    buffer[idx + width - 1] += error * 3.0f / 16.0f;
                if (x + 1 < width)
                    buffer[idx + width + 1] += error * 1.0f / 16.0f;
            }
        }
    }
    delete[] buffer;
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

    unsigned char* halftoned_buffer = new unsigned char[width * height * 4];
    halftone(gray_buffer, halftoned_buffer, width, height);
    result = stbi_write_png("res/textures/Halftone.png", width*2, height*2, 1, halftoned_buffer, width*2);
    std::cout << "Halftone result: " << result << std::endl;

    unsigned char* floyd_steinberg_buffer = new unsigned char[width * height];
    floyd_steinberg(gray_buffer, floyd_steinberg_buffer, width, height);
    result = stbi_write_png("res/textures/FloyedSteinberg.png", width, height, 1, floyd_steinberg_buffer, width);
    std::cout << "FloyedSteinberg result: " << result << std::endl;

    stbi_image_free(buffer);
    stbi_image_free(gray_buffer);
    stbi_image_free(smooth_buffer);
    stbi_image_free(halftoned_buffer);
    stbi_image_free(floyd_steinberg_buffer);
    return 0;
}