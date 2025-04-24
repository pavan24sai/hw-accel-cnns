#pragma once

#ifndef TENSOR3D_H
#define TENSOR3D_H

#include <vector>
#include <iostream>
#include <algorithm>

/*
* This class represents a three-dimensional array of floating-point values, serving as the data structure for neural network operations. 
* It manages feature maps, weights, and intermediate activations with dimensions: depth (channels), height, and width. 
* The class provides access methods and handles the underlying data storage in a contiguous memory layout.
*/
class Tensor3D {
private:
    int depth;
    int height;
    int width;
    std::vector<float> data;

public:
    Tensor3D(int d, int h, int w, float initVal = 0.0f);

    float& at(int d, int h, int w);
    const float& at(int d, int h, int w) const;

    int getDepth() const;
    int getHeight() const;
    int getWidth() const;
    const std::vector<float>& getData() const;
    std::vector<float>& getData();

    void print(int d, int maxH = 5, int maxW = 5) const;
};

#endif // TENSOR3D_H
