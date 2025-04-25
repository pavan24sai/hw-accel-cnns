#include "Tensor3D.h"

Tensor3D::Tensor3D(int d, int h, int w, float initVal) :
    depth(d), height(h), width(w), data(d * h * w, initVal) {}

float& Tensor3D::at(int d, int h, int w) {
    return data[static_cast<size_t>(d) * height * width + h * width + w];
}

const float& Tensor3D::at(int d, int h, int w) const {
    return data[static_cast<size_t>(d) * height * width + h * width + w];
}

int Tensor3D::getDepth() const { return depth; }
int Tensor3D::getHeight() const { return height; }
int Tensor3D::getWidth() const { return width; }
const std::vector<float>& Tensor3D::getData() const { return data; }
std::vector<float>& Tensor3D::getData() { return data; }

void Tensor3D::print(int d, int maxH, int maxW) const {
    std::cout << "Tensor slice for depth " << d << ":" << std::endl;
    for (int h = 0; h < std::min(height, maxH); ++h) {
        for (int w = 0; w < std::min(width, maxW); ++w) {
            std::cout << at(d, h, w) << " ";
        }
        std::cout << std::endl;
    }
}
