#ifndef CNN_PARAMS_H
#define CNN_PARAMS_H

// CNN parameters matching the HLS implementation
#define IMAGE_SIZE 28
#define IMAGE_CHANNELS 1
#define CONV1_KERNEL_SIZE 3
#define CONV1_STRIDE 1
#define CONV1_CHANNELS 1
#define CONV1_FILTERS 32
#define CONV1_BIAS_SIZE 32

#define P1_KERNEL_SIZE 2
#define P1_STRIDE 2
#define P1_SIZE 14

#define CONV2_KERNEL_SIZE 3
#define CONV2_STRIDE 1
#define CONV2_CHANNELS 32
#define CONV2_FILTERS 64
#define CONV2_BIAS_SIZE 64

#define P2_SIZE 7

#define FC1_WEIGHTS_H 3136  // 7 * 7 * 64
#define FC1_WEIGHTS_W 128
#define FC1_BIAS_SIZE 128

#define FC2_WEIGHTS_H 128
#define FC2_WEIGHTS_W 10
#define FC2_BIAS_SIZE 10

// Fashion-MNIST class names
extern const char* FASHION_CLASSES[10];

// Accelerator control register offsets
#define AP_CTRL_REG_OFFSET     0x00
#define AP_DONE_BIT            0x2
#define AP_IDLE_BIT            0x4
#define AP_READY_BIT           0x1
#define AP_START_BIT           0x1

// Input data pointer registers
#define IMAGE_ADDR_OFFSET        0x10
#define CONV1_WEIGHTS_ADDR_OFFSET 0x18
#define CONV1_BIAS_ADDR_OFFSET    0x20
#define CONV2_WEIGHTS_ADDR_OFFSET 0x28
#define CONV2_BIAS_ADDR_OFFSET    0x30
#define FC1_WEIGHTS_ADDR_OFFSET   0x38
#define FC1_BIAS_ADDR_OFFSET      0x40
#define FC2_WEIGHTS_ADDR_OFFSET   0x48
#define FC2_BIAS_ADDR_OFFSET      0x50
#define PREDICTIONS_ADDR_OFFSET   0x58

#endif // CNN_PARAMS_H
