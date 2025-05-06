#ifndef CNN_TYPES_H
#define CNN_TYPES_H

#include <ap_int.h>
#include <ap_fixed.h>
#include "cnn_params.h"

// Fixed-point data type optimized for resource usage
typedef ap_fixed<12, 6> data_t;

// Layer configuration structure
typedef struct {
    int input_channels;   // N
    int output_channels;  // M
    int input_height;     // Input H
    int input_width;      // Input W
    int output_height;    // Output H (R)
    int output_width;     // Output W (C)
    int kernel_size;      // K
    int stride;           // S
    int padding;          // P
} LayerConfig;

#endif // CNN_TYPES_H