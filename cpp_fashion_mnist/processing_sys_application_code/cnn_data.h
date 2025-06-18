#ifndef CNN_DATA_H
#define CNN_DATA_H

#include "cnn_params.h"

#define NUM_TEST_SAMPLES 100

// Weight arrays
extern float conv1_weights[];
extern float conv1_bias[];
extern float conv2_weights[];
extern float conv2_bias[];
extern float fc1_weights[];
extern float fc1_bias[];
extern float fc2_weights[];
extern float fc2_bias[];

// Test dataset info for accuracy evaluation
typedef struct {
    float* image_data;
    int label;
    const char* class_name;
} TestSample;

// Number of test samples
extern TestSample test_samples[];

#endif // CNN_DATA_H
