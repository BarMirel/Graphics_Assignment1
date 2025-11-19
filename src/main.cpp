#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cstring>
#include <vector>
using namespace std;

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

    const float step = 255.0f / 15.0f; //Gives 16 levels of intensity, using / 16 will give an extra level because of the rounding later

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

void canny_gradient(const unsigned char* input, unsigned char* magnitude_output, float* direction_output, int width, int height) {
    int kernelX[3] = { -1, 0, 1 };
    int kernelY[3] = { -1, 0, 1 };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int gx = 0;
            int gy = 0;
            for (int kx = -1; kx <= 1; ++kx) {
                int pixelX = x + kx;
                if (pixelX < 0) pixelX = 0;
                if (pixelX >= width) pixelX = width - 1;
                unsigned char pixelValue = input[y * width + pixelX];
                int kernelValue = kernelX[kx + 1];
                gx += pixelValue * kernelValue;
            }

            for (int ky = -1; ky <= 1; ++ky) {
                int pixelY = y + ky;
                if (pixelY < 0) pixelY = 0;
                if (pixelY >= height) pixelY = height - 1;
                unsigned char pixelValue = input[pixelY * width + x];
                int kernelValue = kernelY[ky + 1];
                gy += pixelValue * kernelValue;
            }

            float magnitude = sqrtf(static_cast<float>(gx * gx + gy * gy));
            float direction = atan2f(static_cast<float>(gy), static_cast<float>(gx));

            magnitude_output[y * width + x] = clip(static_cast<int>(magnitude));
            if (direction_output != nullptr) {
                direction_output[y * width + x] = direction;
            }
        }
    }
}

void non_max_suppression(const unsigned char* magnitude_input, const float* direction_input, unsigned char* output, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float angle = direction_input[y * width + x] * 180.0f / static_cast<float>(M_PI);
            if (angle < 0.0f)
                angle += 180.0f;

            int neighborX1 = 0, neighborY1 = 0;
            int neighborX2 = 0, neighborY2 = 0;

            if ((angle >= 0.0f && angle <= 22.5f) || (angle > 157.5f && angle <= 180.0f)) {
                neighborX1 = -1; neighborY1 = 0;
                neighborX2 = 1;  neighborY2 = 0;
            } else if (angle > 22.5f && angle <= 67.5f) {
                neighborX1 = -1; neighborY1 = 1;
                neighborX2 = 1;  neighborY2 = -1;
            } else if (angle > 67.5f && angle <= 112.5f) {
                neighborX1 = 0; neighborY1 = -1;
                neighborX2 = 0; neighborY2 = 1;
            } else {
                neighborX1 = -1; neighborY1 = -1;
                neighborX2 = 1;  neighborY2 = 1;
            }

            int neighbor1X = x + neighborX1;
            int neighbor1Y = y + neighborY1;
            if (neighbor1X < 0) neighbor1X = 0;
            if (neighbor1X >= width) neighbor1X = width - 1;
            if (neighbor1Y < 0) neighbor1Y = 0;
            if (neighbor1Y >= height) neighbor1Y = height - 1;

            int neighbor2X = x + neighborX2;
            int neighbor2Y = y + neighborY2;
            if (neighbor2X < 0) neighbor2X = 0;
            if (neighbor2X >= width) neighbor2X = width - 1;
            if (neighbor2Y < 0) neighbor2Y = 0;
            if (neighbor2Y >= height) neighbor2Y = height - 1;

            unsigned char currentMagnitude = magnitude_input[y * width + x];
            unsigned char neighborMagnitude1 = magnitude_input[neighbor1Y * width + neighbor1X];
            unsigned char neighborMagnitude2 = magnitude_input[neighbor2Y * width + neighbor2X];

            if (currentMagnitude >= neighborMagnitude1 && currentMagnitude >= neighborMagnitude2) {
                output[y * width + x] = currentMagnitude;
            } else {
                output[y * width + x] = 0;
            }
        }
    }
}

unsigned char find_max_value(const unsigned char* input, int width, int height) {
    unsigned char max_value = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char value = input[y * width + x];
            if (value > max_value) {
                max_value = value;
            }
        }
    }
    return max_value;
}
void double_threshold(const unsigned char* input, unsigned char* output, int width, int height, float low_ratio, float high_ratio) {
    unsigned char max_pixel = find_max_value(input, width, height);
    unsigned char low_threshold = max_pixel * 0.15;
    unsigned char high_threshold = max_pixel * 0.25;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char value = input[y * width + x];
            if (value >= high_threshold) {
                output[y * width + x] = 255;
            } else if (value < low_threshold) {
                output[y * width + x] = 0;  
            } else {
                output[y * width + x] = value; 
            }
        }
    }
}

void hysteresis(const unsigned char* input,unsigned char* output, int width, int height) {
    const int size = width * height;
    for (int i = 0; i < width * height; ++i)
        output[i] = input[i];
    vector<int> stack;
    for (int i = 0; i < size; ++i) {
        if (output[i] == 255) {
            stack.push_back(i);
        }
    }
    //DFS
    while (!stack.empty()) {
        int idx = stack.back();
        stack.pop_back();

        int x = idx % width;
        int y = idx / width;

        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                if (kx == 0 && ky == 0) continue;

                int nx = x + kx;
                int ny = y + ky;
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

                int nidx = ny * width + nx;
                if (output[nidx] != 255 && output[nidx] != 0) {
                    output[nidx] = 255;
                    stack.push_back(nidx); //put new strong in stack to continue DFS
                }
            }
        }
    }

    for (int i = 0; i < size; ++i) {
        if (output[i] != 255) {
            output[i] = 0;
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
    
    //result = stbi_write_png("res/textures/Gaussian.png", width, height, 1, smooth_buffer, width);
    //std::cout << "Gaussian result: " << result << std::endl;

    unsigned char* halftoned_buffer = new unsigned char[width * height * 4];
    halftone(gray_buffer, halftoned_buffer, width, height);
    result = stbi_write_png("res/textures/Halftone.png", width*2, height*2, 1, halftoned_buffer, width*2);
    std::cout << "Halftone result: " << result << std::endl;

    unsigned char* floyd_steinberg_buffer = new unsigned char[width * height];
    floyd_steinberg(gray_buffer, floyd_steinberg_buffer, width, height);
    result = stbi_write_png("res/textures/FloyedSteinberg.png", width, height, 1, floyd_steinberg_buffer, width);
    std::cout << "FloyedSteinberg result: " << result << std::endl;

    unsigned char* gradient_magnitude_buffer = new unsigned char[width * height];
    float* gradient_direction_buffer = new float[width * height];
    canny_gradient(smooth_buffer, gradient_magnitude_buffer, gradient_direction_buffer, width, height);
    //result = stbi_write_png("res/textures/Gradient.png", width, height, 1, gradient_magnitude_buffer, width);
    //std::cout << "Gradient result: " << result << std::endl;

    unsigned char* non_max_buffer = new unsigned char[width * height];
    non_max_suppression(gradient_magnitude_buffer, gradient_direction_buffer, non_max_buffer, width, height);
    //result = stbi_write_png("res/textures/NonMax.png", width, height, 1, non_max_buffer, width);
    //std::cout << "NonMax result: " << result << std::endl;

    unsigned char* double_threshold_buffer = new unsigned char[width * height];
    double_threshold(non_max_buffer, double_threshold_buffer, width, height, 0.05f, 0.25f);
    //result = stbi_write_png("res/textures/DoubleThreshold.png", width, height, 1, double_threshold_buffer, width);
    //std::cout << "DoubleThreshold result: " << result << std::endl;

    unsigned char* hysteresis_buffer = new unsigned char[width * height];
    hysteresis(double_threshold_buffer, hysteresis_buffer, width, height);
    //result = stbi_write_png("res/textures/Hysteresis.png", width, height, 1, hysteresis_buffer, width);
    //std::cout << "Hysteresis result: " << result << std::endl;

    result = stbi_write_png("res/textures/Canny.png", width, height, 1, hysteresis_buffer, width);
    std::cout << "Canny result: " << result << std::endl;

    stbi_image_free(buffer);
    stbi_image_free(gray_buffer);
    stbi_image_free(smooth_buffer);
    stbi_image_free(halftoned_buffer);
    stbi_image_free(floyd_steinberg_buffer);
    stbi_image_free(gradient_magnitude_buffer);
    stbi_image_free(non_max_buffer);
    stbi_image_free(double_threshold_buffer);
    stbi_image_free(hysteresis_buffer);
    delete[] gradient_direction_buffer;
    return 0;
}