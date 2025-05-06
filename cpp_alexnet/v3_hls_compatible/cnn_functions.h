#ifndef CNN_FUNCTIONS_H
#define CNN_FUNCTIONS_H

#include "cnn_types.h"

// Memory transfer functions
void load_input_tile(
    data_t* input_ddr,
    data_t input_buffer[TN][INPUT_TILE_HEIGHT][INPUT_TILE_WIDTH],
    int n_offset, int h_offset, int w_offset,
    int N, int H, int W, int S, int P);

void load_weight_tile(
    data_t* weights_ddr,
    data_t weight_buffer[TM][TN][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE],
    int m_offset, int n_offset,
    int M, int N, int K);

void load_bias(
    data_t* bias_ddr,
    data_t bias_buffer[TM],
    int m_offset, int M);

void init_output_buffer(
    data_t output_buffer[TM][TR][TC],
    data_t bias_buffer[TM],
    int tm_bound);

void store_output_tile(
    data_t* output_ddr,
    data_t output_buffer[TM][TR][TC],
    int m_offset, int h_offset, int w_offset,
    int M, int R, int C);

// Compute engine functions
void compute_tile(
    data_t input_buffer[TN][INPUT_TILE_HEIGHT][INPUT_TILE_WIDTH],
    data_t weight_buffer[TM][TN][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE],
    data_t output_buffer[TM][TR][TC],
    int kernel_size, int stride, int tm_bound, int tn_bound, int tr_bound, int tc_bound);

void apply_relu(
    data_t buffer[TM][TR][TC], 
    int tm, int tr, int tc);

// Top-level accelerator function
void fashion_mnist_cnn_accelerator(
    data_t* input_ddr,
    data_t* output_ddr,
    data_t* weights_ddr,
    data_t* bias_ddr,
    LayerConfig layer_config,
    int layer_idx);

#endif // CNN_FUNCTIONS_H