#include "cnn_functions.h"

// Top-level accelerator function
void fashion_mnist_cnn_accelerator(
    data_t* input_ddr,      // Input feature maps in DDR
    data_t* output_ddr,     // Output feature maps in DDR
    data_t* weights_ddr,    // Weights in DDR
    data_t* bias_ddr,       // Bias values in DDR
    LayerConfig layer_config,// Layer configuration
    int layer_idx           // Current layer index
) {
    // Specify interface types with optimized parameters for KV260
    #pragma HLS INTERFACE m_axi port=input_ddr offset=slave bundle=INPUT_AXI depth=TEST_MAX_INPUT_SIZE max_read_burst_length=8 max_write_burst_length=8
    #pragma HLS INTERFACE m_axi port=output_ddr offset=slave bundle=OUTPUT_AXI depth=TEST_MAX_OUTPUT_SIZE max_read_burst_length=8 max_write_burst_length=8
    #pragma HLS INTERFACE m_axi port=weights_ddr offset=slave bundle=WEIGHTS_AXI depth=TEST_MAX_WEIGHT_SIZE max_read_burst_length=8 max_write_burst_length=8
    #pragma HLS INTERFACE m_axi port=bias_ddr offset=slave bundle=BIAS_AXI depth=TEST_MAX_BIAS_SIZE max_read_burst_length=8 max_write_burst_length=8
    #pragma HLS INTERFACE s_axilite port=layer_config bundle=CONTROL
    #pragma HLS INTERFACE s_axilite port=layer_idx bundle=CONTROL
    #pragma HLS INTERFACE s_axilite port=return bundle=CONTROL
    
    // Extract layer parameters
    int N = layer_config.input_channels;
    int M = layer_config.output_channels;
    int input_H = layer_config.input_height;
    int input_W = layer_config.input_width;
    int output_H = layer_config.output_height;
    int output_W = layer_config.output_width;
    int K = layer_config.kernel_size;
    int S = layer_config.stride;
    int P = layer_config.padding;
    
    // On-chip buffers with balanced partitioning for KV260
    data_t input_buffer[TN][INPUT_TILE_HEIGHT][INPUT_TILE_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=input_buffer dim=1 complete
    
    data_t weight_buffer[TM][TN][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE];
    #pragma HLS ARRAY_PARTITION variable=weight_buffer dim=1 cyclic factor=2
    #pragma HLS ARRAY_PARTITION variable=weight_buffer dim=2 cyclic factor=2
    
    data_t output_buffer[TM][TR][TC];
    #pragma HLS ARRAY_PARTITION variable=output_buffer dim=1 cyclic factor=2
    
    data_t bias_buffer[TM];
    #pragma HLS ARRAY_PARTITION variable=bias_buffer cyclic factor=2
    
    // Processing logic - single tile case
    if (N <= TN && M <= TM && output_H <= TR && output_W <= TC) {
        load_bias(bias_ddr, bias_buffer, 0, M);
        load_weight_tile(weights_ddr, weight_buffer, 0, 0, M, N, K);
        load_input_tile(input_ddr, input_buffer, 0, 0, 0, N, input_H, input_W, S, P);
        init_output_buffer(output_buffer, bias_buffer, M);
        compute_tile(input_buffer, weight_buffer, output_buffer, K, S, M, N, output_H, output_W);
        apply_relu(output_buffer, M, output_H, output_W);
        store_output_tile(output_ddr, output_buffer, 0, 0, 0, M, output_H, output_W);
    }
    else {
        // General tiled processing 
        int tm_steps = (M + TM - 1) / TM;
        int tn_steps = (N + TN - 1) / TN;
        int tr_steps = (output_H + TR - 1) / TR;
        int tc_steps = (output_W + TC - 1) / TC;
        
        // Implement tiling following the paper's approach
        tm_loop: for (int tm = 0; tm < tm_steps; tm++) {
            int m_offset = tm * TM;
            int tm_bound = (M - m_offset < TM) ? (M - m_offset) : TM;
            load_bias(bias_ddr, bias_buffer, m_offset, M);
            
            tr_loop: for (int tr = 0; tr < tr_steps; tr++) {
                int r_offset = tr * TR;
                int tr_bound = (output_H - r_offset < TR) ? (output_H - r_offset) : TR;
                
                tc_loop: for (int tc = 0; tc < tc_steps; tc++) {
                    int c_offset = tc * TC;
                    int tc_bound = (output_W - c_offset < TC) ? (output_W - c_offset) : TC;
                    
                    // Initialize output with bias
                    init_output_buffer(output_buffer, bias_buffer, tm_bound);
                    
                    // Process input channel tiles
                    tn_loop: for (int tn = 0; tn < tn_steps; tn++) {
                        int n_offset = tn * TN;
                        int tn_bound = (N - n_offset < TN) ? (N - n_offset) : TN;
                        
                        // Load data for current tile
                        load_weight_tile(weights_ddr, weight_buffer, m_offset, n_offset, M, N, K);
                        load_input_tile(input_ddr, input_buffer, n_offset, r_offset, c_offset, N, input_H, input_W, S, P);
                        
                        // Compute convolution
                        compute_tile(input_buffer, weight_buffer, output_buffer, K, S, tm_bound, tn_bound, tr_bound, tc_bound);
                    }
                    
                    // Apply ReLU and store output
                    apply_relu(output_buffer, tm_bound, tr_bound, tc_bound);
                    store_output_tile(output_ddr, output_buffer, m_offset, r_offset, c_offset, M, output_H, output_W);
                }
            }
        }
    }
}