#pragma once

#ifndef UTILS_H
#define UTILS_H

#include "Tensor3D.h"
#include <string>

// Utility functions for image loading and preprocessing
Tensor3D loadImage(const std::string& filename, int targetHeight, int targetWidth);
Tensor3D loadAndPreprocessImage(const std::string& filename, int targetHeight, int targetWidth);

#endif // UTILS_H
#pragma once
