/******************************************************************************
 * Fashion MNIST CNN - FIXED FOR C/RTL CO-SIMULATION
 * - Separated AXI interface bundles to prevent memory corruption
 * - Fixed buffer management and array indexing
 ******************************************************************************/

#include "headers/defines.h"
#include "headers/activations.h"
#include <hls_math.h>

// DEBUG MACRO - extensive debugging for C/RTL co-simulation
// Disable all debug prints unconditionally
#ifdef DEBUG_PRINT
    #undef DEBUG_PRINT
#endif
#define DEBUG_PRINT(x)

#ifdef DEBUG_ARRAY
    #undef DEBUG_ARRAY
#endif
#define DEBUG_ARRAY(name, arr, size, max_print)

// Simple inline functions for index calculation
inline int conv1_weight_idx(int c_out, int c_in, int h, int w) {
    #pragma HLS INLINE
    return h * (CONV1_KERNEL_SIZE * CONV1_CHANNELS * CONV1_FILTERS) +
           w * (CONV1_CHANNELS * CONV1_FILTERS) +
           c_in * CONV1_FILTERS +
           c_out;
}

inline int conv2_weight_idx(int c_out, int c_in, int h, int w) {
    #pragma HLS INLINE
    return h * (CONV2_KERNEL_SIZE * CONV1_FILTERS * CONV2_FILTERS) +
           w * (CONV1_FILTERS * CONV2_FILTERS) +
           c_in * CONV2_FILTERS +
           c_out;
}

inline int get_3d_index(int ch, int row, int col, int height, int width) {
    #pragma HLS INLINE
    return ch * height * width + row * width + col;
}

// CONV1 with extensive debugging
void conv_layer1_minimal(
    float24_t* output,
    const float24_t* input,
    const float24_t* weights,
    const float24_t* bias) {
    
    #pragma HLS INLINE off
    
    DEBUG_PRINT("[DEBUG] Starting conv_layer1_minimal");
    
    const int padding = 1;
    
    CONV1_FILTER: for (int of = 0; of < CONV1_FILTERS; of++) {
        CONV1_ROW: for (int row = 0; row < IMAGE_SIZE; row++) {
            #pragma HLS LOOP_TRIPCOUNT min=28 max=28
            
            CONV1_COL: for (int col = 0; col < IMAGE_SIZE; col++) {
                #pragma HLS PIPELINE II=9
                
                float24_t sum = bias[of];
                
                for (int ki = 0; ki < CONV1_KERNEL_SIZE; ki++) {
                    for (int kj = 0; kj < CONV1_KERNEL_SIZE; kj++) {
                        for (int if_idx = 0; if_idx < CONV1_CHANNELS; if_idx++) {
                            int in_row = row + ki - padding;
                            int in_col = col + kj - padding;
                            
                            float24_t input_val = float24_t(0);
                            if (in_row >= 0 && in_row < IMAGE_SIZE && 
                                in_col >= 0 && in_col < IMAGE_SIZE) {
                                input_val = input[get_3d_index(if_idx, in_row, in_col, IMAGE_SIZE, IMAGE_SIZE)];
                            }
                            
                            int weight_idx = conv1_weight_idx(of, if_idx, ki, kj);
                            sum += input_val * weights[weight_idx];
                        }
                    }
                }
                
                output[get_3d_index(of, row, col, IMAGE_SIZE, IMAGE_SIZE)] = relu(sum);
            }
        }
    }
    
    DEBUG_PRINT("[DEBUG] Finished conv_layer1_minimal");
}

void pool_layer1_minimal(
    float24_t* output,
    const float24_t* input) {
    
    #pragma HLS INLINE off
    
    DEBUG_PRINT("[DEBUG] Starting pool_layer1_minimal");
    DEBUG_ARRAY("POOL1 input", input, CONV1_FILTERS * IMAGE_SIZE * IMAGE_SIZE, 5);
    
    POOL1_CH: for (int ch = 0; ch < CONV1_FILTERS; ch++) {
        POOL1_ROW: for (int row = 0; row < P1_SIZE; row++) {
            POOL1_COL: for (int col = 0; col < P1_SIZE; col++) {
                #pragma HLS PIPELINE II=1
                
                int start_row = row * 2;
                int start_col = col * 2;
                
                float24_t val0 = input[get_3d_index(ch, start_row, start_col, IMAGE_SIZE, IMAGE_SIZE)];
                float24_t val1 = input[get_3d_index(ch, start_row, start_col + 1, IMAGE_SIZE, IMAGE_SIZE)];
                float24_t val2 = input[get_3d_index(ch, start_row + 1, start_col, IMAGE_SIZE, IMAGE_SIZE)];
                float24_t val3 = input[get_3d_index(ch, start_row + 1, start_col + 1, IMAGE_SIZE, IMAGE_SIZE)];
                
                float24_t max_val = max_pool(max_pool(val0, val1), max_pool(val2, val3));
                output[get_3d_index(ch, row, col, P1_SIZE, P1_SIZE)] = max_val;
            }
        }
    }
    
    DEBUG_ARRAY("POOL1 output", output, CONV1_FILTERS * P1_SIZE * P1_SIZE, 5);
    DEBUG_PRINT("[DEBUG] Finished pool_layer1_minimal");
}

void conv_layer2_minimal(
    float24_t* output,
    const float24_t* input,
    const float24_t* weights,
    const float24_t* bias) {
    
    #pragma HLS INLINE off
    
    DEBUG_PRINT("[DEBUG] Starting conv_layer2_minimal");
    
    const int padding = 1;
    
    CONV2_FILTER: for (int of = 0; of < CONV2_FILTERS; of++) {
        CONV2_ROW: for (int row = 0; row < P1_SIZE; row++) {
            CONV2_COL: for (int col = 0; col < P1_SIZE; col++) {
                #pragma HLS PIPELINE II=12
                
                float24_t sum = bias[of];
                
                for (int if_idx = 0; if_idx < CONV1_FILTERS; if_idx++) {
                    for (int ki = 0; ki < CONV2_KERNEL_SIZE; ki++) {
                        for (int kj = 0; kj < CONV2_KERNEL_SIZE; kj++) {
                            int in_row = row + ki - padding;
                            int in_col = col + kj - padding;
                            
                            float24_t input_val = float24_t(0);
                            if (in_row >= 0 && in_row < P1_SIZE && 
                                in_col >= 0 && in_col < P1_SIZE) {
                                input_val = input[get_3d_index(if_idx, in_row, in_col, P1_SIZE, P1_SIZE)];
                            }
                            
                            int weight_idx = conv2_weight_idx(of, if_idx, ki, kj);
                            sum += input_val * weights[weight_idx];
                        }
                    }
                }
                
                output[get_3d_index(of, row, col, P1_SIZE, P1_SIZE)] = relu(sum);
            }
        }
    }
    
    DEBUG_PRINT("[DEBUG] Finished conv_layer2_minimal");
}

void pool_layer2_minimal(
    float24_t* output,
    const float24_t* input) {
    
    #pragma HLS INLINE off
    
    DEBUG_PRINT("[DEBUG] Starting pool_layer2_minimal");
    DEBUG_ARRAY("POOL2 input", input, CONV2_FILTERS * P1_SIZE * P1_SIZE, 5);
    
    POOL2_CH: for (int ch = 0; ch < CONV2_FILTERS; ch++) {
        POOL2_ROW: for (int row = 0; row < P2_SIZE; row++) {
            POOL2_COL: for (int col = 0; col < P2_SIZE; col++) {
                #pragma HLS PIPELINE II=1
                
                int start_row = row * 2;
                int start_col = col * 2;
                
                float24_t val0 = input[get_3d_index(ch, start_row, start_col, P1_SIZE, P1_SIZE)];
                float24_t val1 = input[get_3d_index(ch, start_row, start_col + 1, P1_SIZE, P1_SIZE)];
                float24_t val2 = input[get_3d_index(ch, start_row + 1, start_col, P1_SIZE, P1_SIZE)];
                float24_t val3 = input[get_3d_index(ch, start_row + 1, start_col + 1, P1_SIZE, P1_SIZE)];
                
                float24_t max_val = max_pool(max_pool(val0, val1), max_pool(val2, val3));
                output[get_3d_index(ch, row, col, P2_SIZE, P2_SIZE)] = max_val;
            }
        }
    }
    
    DEBUG_ARRAY("POOL2 output", output, CONV2_FILTERS * P2_SIZE * P2_SIZE, 5);
    DEBUG_PRINT("[DEBUG] Finished pool_layer2_minimal");
}

void fc_layer1_minimal(
    float24_t* output,
    const float24_t* input,
    const float24_t* weights,
    const float24_t* bias) {
    
    #pragma HLS INLINE off
    
    DEBUG_PRINT("[DEBUG] Starting fc_layer1_minimal");
    DEBUG_ARRAY("FC1 input", input, FC1_WEIGHTS_H, 10);
    
    // Initialize output with bias
    for (int j = 0; j < FC1_WEIGHTS_W; j++) {
        output[j] = bias[j];
    }
    
    // Process one output at a time - CLEAN AND SIMPLE
    for (int j = 0; j < FC1_WEIGHTS_W; j++) {
        float24_t acc = float24_t(0);
        
        for (int i = 0; i < FC1_WEIGHTS_H; i++) {
            #pragma HLS PIPELINE II=1
            acc += input[i] * weights[i * FC1_WEIGHTS_W + j];
        }
        
        output[j] += acc;
        output[j] = relu(output[j]);
    }
    
    DEBUG_ARRAY("FC1 output", output, FC1_WEIGHTS_W, 10);
    DEBUG_PRINT("[DEBUG] Finished fc_layer1_minimal");
}

void flatten_minimal(
    float24_t* output,
    const float24_t* input) {
    
    #pragma HLS INLINE off
    
    // Original flattening logic that works in C simulation
    FLATTEN_LOOP: for (int i = 0; i < FC1_WEIGHTS_H; i++) {
        int ch = i % CONV2_FILTERS;
        int spatial_idx = i / CONV2_FILTERS;
        int row = spatial_idx / P2_SIZE;
        int col = spatial_idx % P2_SIZE;
        output[i] = input[get_3d_index(ch, row, col, P2_SIZE, P2_SIZE)];
    }
}

void fc_layer2_minimal(
    float24_t* output,
    const float24_t* input,
    const float24_t* weights,
    const float24_t* bias) {
    
    #pragma HLS INLINE off
    
    DEBUG_PRINT("[DEBUG] Starting fc_layer2_minimal");
    DEBUG_ARRAY("FC2 bias", bias, FC2_WEIGHTS_W, 10);
    DEBUG_ARRAY("FC2 input", input, FC1_WEIGHTS_W, 10);
    
    float24_t raw_output[FC2_WEIGHTS_W];
    
    // Initialize with bias
    for (int j = 0; j < FC2_WEIGHTS_W; j++) {
        raw_output[j] = bias[j];
    }
    
    // Process one output at a time
    for (int j = 0; j < FC2_WEIGHTS_W; j++) {
        float24_t acc = float24_t(0);
        
        for (int i = 0; i < FC1_WEIGHTS_W; i++) {
            #pragma HLS PIPELINE II=1
            acc += input[i] * weights[i * FC2_WEIGHTS_W + j];
        }
        
        raw_output[j] += acc;
    }
    
    DEBUG_ARRAY("FC2 raw_output", raw_output, FC2_WEIGHTS_W, 10);
    
    // Modified softmax with proper type handling
    float24_t max_val = raw_output[0];
    for (int j = 1; j < FC2_WEIGHTS_W; j++) {
        if (raw_output[j] > max_val) max_val = raw_output[j];
    }
    
    float24_t sum_exp = float24_t(0);
    float24_t exp_vals[FC2_WEIGHTS_W];
    
    for (int j = 0; j < FC2_WEIGHTS_W; j++) {
        float diff = (float)(raw_output[j] - max_val);
        exp_vals[j] = hls::exp(diff);
        sum_exp += exp_vals[j];
    }
    
    // Proper comparison with float24_t type
    if (sum_exp != float24_t(0)) {
        for (int j = 0; j < FC2_WEIGHTS_W; j++) {
            output[j] = exp_vals[j] / sum_exp;
        }
    } else {
        // Handle division by zero
        for (int j = 0; j < FC2_WEIGHTS_W; j++) {
            output[j] = (j == 0) ? float24_t(1.0) : float24_t(0.0);
        }
    }
    
    DEBUG_ARRAY("FC2 final output", output, FC2_WEIGHTS_W, 10);
    DEBUG_PRINT("[DEBUG] Finished fc_layer2_minimal");
}

void nnet(
    float24_t* image,
    float24_t* conv1_weights,
    float24_t* conv1_bias,
    float24_t* conv2_weights,
    float24_t* conv2_bias,
    float24_t* fc1_weights,
    float24_t* fc1_bias,
    float24_t* fc2_weights,
    float24_t* fc2_bias,
    float24_t* predictions
) {
    // MAXI interfaces with proper alignment and burst lengths
    #pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0 depth=784 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=conv1_weights offset=slave bundle=gmem1 depth=288 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=conv1_bias offset=slave bundle=gmem2 depth=32 max_read_burst_length=16
    #pragma HLS INTERFACE m_axi port=conv2_weights offset=slave bundle=gmem3 depth=18432 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=conv2_bias offset=slave bundle=gmem4 depth=64 max_read_burst_length=16
    #pragma HLS INTERFACE m_axi port=fc1_weights offset=slave bundle=gmem5 depth=401408 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=fc1_bias offset=slave bundle=gmem6 depth=128 max_read_burst_length=16
    #pragma HLS INTERFACE m_axi port=fc2_weights offset=slave bundle=gmem7 depth=1280 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=fc2_bias offset=slave bundle=gmem8 depth=10 max_read_burst_length=16
    #pragma HLS INTERFACE m_axi port=predictions offset=slave bundle=gmem9 depth=10 max_write_burst_length=16
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    // Local copies of bias values
    float24_t local_conv1_bias[CONV1_FILTERS];
    float24_t local_conv2_bias[CONV2_FILTERS];
    float24_t local_fc1_bias[FC1_WEIGHTS_W];
    float24_t local_fc2_bias[FC2_WEIGHTS_W];
    float24_t local_predictions[FC2_WEIGHTS_W];
    
    // Copy bias values locally
    for (int i = 0; i < CONV1_FILTERS; i++) {
        local_conv1_bias[i] = conv1_bias[i];
    }
    
    for (int i = 0; i < CONV2_FILTERS; i++) {
        local_conv2_bias[i] = conv2_bias[i];
    }
    
    for (int i = 0; i < FC1_WEIGHTS_W; i++) {
        local_fc1_bias[i] = fc1_bias[i];
    }
    
    for (int i = 0; i < FC2_WEIGHTS_W; i++) {
        local_fc2_bias[i] = fc2_bias[i];
    }
    
    // Initialize predictions
    for (int i = 0; i < FC2_WEIGHTS_W; i++) {
        local_predictions[i] = float24_t(0);
    }
    
    // Intermediate buffers
    static float24_t conv1_out[CONV1_FILTERS * IMAGE_SIZE * IMAGE_SIZE];
    static float24_t pool1_out[CONV1_FILTERS * P1_SIZE * P1_SIZE];
    static float24_t conv2_out[CONV2_FILTERS * P1_SIZE * P1_SIZE];
    static float24_t pool2_out[CONV2_FILTERS * P2_SIZE * P2_SIZE];
    static float24_t flattened[FC1_WEIGHTS_H];
    static float24_t fc1_out[FC1_WEIGHTS_W];
    
    DEBUG_PRINT("[DEBUG] Starting nnet function");
    
    // Run network
    conv_layer1_minimal(conv1_out, image, conv1_weights, local_conv1_bias);
    pool_layer1_minimal(pool1_out, conv1_out);
    conv_layer2_minimal(conv2_out, pool1_out, conv2_weights, local_conv2_bias);
    pool_layer2_minimal(pool2_out, conv2_out);
    flatten_minimal(flattened, pool2_out);
    fc_layer1_minimal(fc1_out, flattened, fc1_weights, local_fc1_bias);
    fc_layer2_minimal(local_predictions, fc1_out, fc2_weights, local_fc2_bias);
    
    // Copy results with explicit loop for better burst handling
    for (int i = 0; i < FC2_WEIGHTS_W; i++) {
        #pragma HLS PIPELINE II=1
        predictions[i] = local_predictions[i];
    }
    
    DEBUG_ARRAY("nnet FINAL predictions", local_predictions, FC2_WEIGHTS_W, 10);
    DEBUG_PRINT("[DEBUG] nnet function complete");
}