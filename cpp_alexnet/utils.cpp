#include "utils.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <cmath>

Tensor3D loadImage(const std::string& filename, int targetHeight, int targetWidth) {
    // In a real implementation, this would load and preprocess the image
    // For now, just create a dummy input tensor
    Tensor3D input(3, targetHeight, targetWidth, 0.5f);  // 3 channels for RGB
    std::cout << "Loaded dummy image (in real implementation, would load: " << filename << ")" << std::endl;
    return input;
}

Tensor3D loadAndPreprocessImage(const std::string& filename, int targetHeight, int targetWidth) {
    // In a real implementation, you would use a library like OpenCV to load images
    // For simplicity, this is a basic implementation using stb_image

    int width, height, channels;
    unsigned char* img = stbi_load(filename.c_str(), &width, &height, &channels, 3);

    if (!img) {
        std::cerr << "Error: Could not load image: " << filename << std::endl;
        return Tensor3D(3, targetHeight, targetWidth, 0.0f);
    }

    // Create tensor for image data
    Tensor3D imageTensor(3, targetHeight, targetWidth, 0.0f);

    // Resize and preprocess (simple nearest neighbor for demonstration)
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < targetHeight; h++) {
            for (int w = 0; w < targetWidth; w++) {
                // Map target coordinates to source image
                int srcH = static_cast<int>(static_cast<float>(h) / targetHeight * height);
                int srcW = static_cast<int>(static_cast<float>(w) / targetWidth * width);

                // Get pixel value from source image
                int pixelIdx = (srcH * width + srcW) * 3 + c;
                float pixelValue = static_cast<float>(img[pixelIdx]) / 255.0f;

                // ImageNet normalization (mean and std for each channel)
                if (c == 0) {  // R channel
                    pixelValue = (pixelValue - 0.485f) / 0.229f;
                }
                else if (c == 1) {  // G channel
                    pixelValue = (pixelValue - 0.456f) / 0.224f;
                }
                else {  // B channel
                    pixelValue = (pixelValue - 0.406f) / 0.225f;
                }

                imageTensor.at(c, h, w) = pixelValue;
            }
        }
    }

    // Free image data
    stbi_image_free(img);

    std::cout << "Loaded and preprocessed image: " << filename << " to shape [3, "
        << targetHeight << ", " << targetWidth << "]" << std::endl;

    return imageTensor;
}