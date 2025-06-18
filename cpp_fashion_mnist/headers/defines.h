/******************************************************************************
 * Fashion MNIST CNN Defines - Optimized for AMD Kria KV260
 * Tailored for KV260 FPGA resources with external weight loading
 ******************************************************************************/

#ifndef DEFINES_H
#define DEFINES_H

// Fixed-point precision optimized for KV260 DSP slices and CNN accuracy
// Using 32-bit with 16 integer bits, 16 fractional bits for sufficient range
// This configuration prevents overflow in FC layer accumulations while maintaining precision
#define EXP_WIDTH 16
#define INT_WIDTH 5

// Define the fixed-point type
#include "ap_fixed.h"
typedef ap_fixed<EXP_WIDTH, INT_WIDTH> float24_t;

// Input image parameters
#define IMAGE_SIZE 28
#define IMAGE_CHANNELS 1

// Convolutional Layer 1 parameters
#define CONV1_KERNEL_SIZE 3
#define CONV1_STRIDE 1
#define CONV1_CHANNELS 1
#define CONV1_FILTERS 32
#define CONV1_BIAS_SIZE 32

// Pooling Layer 1 parameters
#define P1_KERNEL_SIZE 2
#define P1_STRIDE 2
#define P1_CHANNELS 32
#define A1_SIZE 28  // Output size of conv1 (same padding)
#define A1_CHANNELS 32
#define P1_SIZE 14  // Output size after pooling

// Convolutional Layer 2 parameters
#define CONV2_KERNEL_SIZE 3
#define CONV2_STRIDE 1
#define CONV2_CHANNELS 32
#define CONV2_FILTERS 64
#define CONV2_BIAS_SIZE 64

// Pooling Layer 2 parameters
#define P2_KERNEL_SIZE 2
#define P2_STRIDE 2
#define P2_CHANNELS 64
#define A2_SIZE 14  // Output size of conv2 (same padding)
#define A2_CHANNELS 64
#define P2_SIZE 7   // Output size after pooling

// Fully Connected Layer 1 parameters
#define FC1_WEIGHTS_H 3136  // 7 * 7 * 64
#define FC1_WEIGHTS_W 128
#define FC1_BIAS_SIZE 128
#define FC1_ACT_SIZE 128

// Fully Connected Layer 2 (Output) parameters
#define FC2_WEIGHTS_H 128
#define FC2_WEIGHTS_W 10
#define FC2_BIAS_SIZE 10
#define FC2_ACT_SIZE 10

// Optimization parameters for AMD Kria KV260
// These are tuned for the enhanced resources of the KV260

// Tiling parameters optimized for KV260 BRAM and URAM usage
#define MAX_TILE_SIZE 128
#define MAX_CHANNEL_TILE 32
#define MAX_KERNEL_UNROLL 9  // 3x3 kernel fully unrolled

// Resource usage thresholds (optimized for KV260)
#define MAX_DSP_USAGE 75      // % of available DSPs
#define MAX_BRAM_USAGE 60     // % of available BRAMs (save for URAM usage)
#define MAX_URAM_USAGE 70     // % of available URAMs
#define MAX_LUT_USAGE 80      // % of available LUTs

// Stream depths optimized for KV260 (enhanced memory)
#define SMALL_STREAM_DEPTH 32
#define MEDIUM_STREAM_DEPTH 128
#define LARGE_STREAM_DEPTH 512

// Loop unrolling factors optimized for KV260 (Chen et al. 2015)
#define CONV_UNROLL_FACTOR 4  // Uniform unroll factor
#define FC_UNROLL_FACTOR 4    // Uniform unroll factor
#define POOL_UNROLL_FACTOR 4  // Uniform unroll factor

// Memory bandwidth optimization for KV260
#define BURST_SIZE 32
#define MAX_OUTSTANDING_READS 16

// Data layout optimization
#define WEIGHT_ALIGNMENT 32
#define DATA_ALIGNMENT 16

// Compiler pragmas for optimization
#define PIPELINE_II_1 1
#define PIPELINE_II_2 2
#define PIPELINE_II_4 4

// Platform-specific configurations for AMD Kria KV260 (Zynq UltraScale+ K26)
#define K26_DSP_COUNT 1248      // DSP48E2 slices
#define K26_BRAM_COUNT 312      // 18Kb BRAM blocks
#define K26_URAM_COUNT 96       // UltraRAM blocks (288Kb each)
#define K26_LUT_COUNT 117120    // CLB LUTs
#define K26_FF_COUNT 234240     // CLB Flip-Flops
#define K26_SLICE_COUNT 17280   // CLB slices

// Calculate available resources (with safety margin for KV260)
#define AVAILABLE_DSP (K26_DSP_COUNT * MAX_DSP_USAGE / 100)
#define AVAILABLE_BRAM (K26_BRAM_COUNT * MAX_BRAM_USAGE / 100)
#define AVAILABLE_URAM (K26_URAM_COUNT * MAX_URAM_USAGE / 100)
#define AVAILABLE_LUT (K26_LUT_COUNT * MAX_LUT_USAGE / 100)

// Derived parameters for optimization
#define MAX_PARALLEL_MACS AVAILABLE_DSP
#define MAX_BRAM_BUFFER_SIZE (AVAILABLE_BRAM * 1024)  // In bytes (18Kb each)
#define MAX_URAM_BUFFER_SIZE (AVAILABLE_URAM * 36864) // In bytes (288Kb each)

// Cache-line optimization for DDR access on KV260
#define DDR_CACHE_LINE_SIZE 64
#define DDR_BURST_LENGTH 16

// AXI interface optimization for KV260
#define AXI_DATA_WIDTH 128     // Enhanced for KV260
#define AXI_ADDR_WIDTH 40      // 40-bit addressing for KV260
#define AXI_ID_WIDTH 6         // Enhanced ID width

// Debug and profiling options
#ifdef DEBUG_MODE
#define DEBUG_PRINT(x) std::cout << x << std::endl
#define DEBUG_ASSERT(cond) assert(cond)
#else
#define DEBUG_PRINT(x)
#define DEBUG_ASSERT(cond)
#endif

// Performance monitoring
#ifdef ENABLE_PROFILING
#define PROFILE_START(name) auto start_##name = std::chrono::high_resolution_clock::now()
#define PROFILE_END(name) auto end_##name = std::chrono::high_resolution_clock::now(); \
                         auto duration_##name = std::chrono::duration_cast<std::chrono::microseconds>(end_##name - start_##name); \
                         std::cout << #name << " took " << duration_##name.count() << " microseconds" << std::endl
#else
#define PROFILE_START(name)
#define PROFILE_END(name)
#endif

// Safety checks for parameter validity on KV260
#if CONV1_FILTERS > MAX_PARALLEL_MACS
#warning "CONV1_FILTERS exceeds available DSP resources on KV260"
#endif

#if CONV2_FILTERS > MAX_PARALLEL_MACS
#warning "CONV2_FILTERS exceeds available DSP resources on KV260"
#endif

#if FC1_WEIGHTS_W > MAX_PARALLEL_MACS
#warning "FC1_WEIGHTS_W may exceed available DSP resources on KV260"
#endif

// Utility macros
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Alignment macros
#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))
#define IS_ALIGNED(x, align) (((x) & ((align) - 1)) == 0)

#endif // DEFINES_H