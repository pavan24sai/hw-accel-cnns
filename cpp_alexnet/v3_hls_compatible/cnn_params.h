#ifndef CNN_PARAMS_H
#define CNN_PARAMS_H

// Optimized tile sizes for Kria KV260
#define TM 8               // Tile size for output channels
#define TN 4               // Tile size for input channels
#define TR 7               // Tile size for output rows
#define TC 7               // Tile size for output columns

// Maximum parameters - optimized for MNIST CNN
#define MAX_KERNEL_SIZE 5  // Most CNN kernels are 3x3 or 5x5
#define MAX_STRIDE 2       // MNIST typically uses stride 1 or 2

// Derived parameters for buffer sizes
#define INPUT_TILE_HEIGHT (TR*MAX_STRIDE + MAX_KERNEL_SIZE - MAX_STRIDE)
#define INPUT_TILE_WIDTH (TC*MAX_STRIDE + MAX_KERNEL_SIZE - MAX_STRIDE)

// Memory interface parameters for KV260
#define DDR_INTERFACE_WIDTH 128
#define AXI_BURST_LEN 8

// Test parameters for co-simulation
#define TEST_MAX_INPUT_SIZE 512
#define TEST_MAX_OUTPUT_SIZE 512
#define TEST_MAX_WEIGHT_SIZE 1024
#define TEST_MAX_BIAS_SIZE 32

#endif // CNN_PARAMS_H