#include "cnn_functions.h"

// Function to load input feature map from DDR to on-chip buffer
void load_input_tile(
    data_t* input_ddr,
    data_t input_buffer[TN][INPUT_TILE_HEIGHT][INPUT_TILE_WIDTH],
    int n_offset, int h_offset, int w_offset,
    int N, int H, int W, int S, int P) {
    
    #pragma HLS INLINE off
    
    // Pre-compute limits to avoid complex conditions
    const int n_limit = ((n_offset + TN) > N) ? (N - n_offset) : TN;
    
    // Clear buffer first (separate loop)
    clear_input: for (int n = 0; n < TN; n++) {
        for (int h = 0; h < INPUT_TILE_HEIGHT; h++) {
            #pragma HLS PIPELINE II=1
            for (int w = 0; w < INPUT_TILE_WIDTH; w++) {
                input_buffer[n][h][w] = 0;
            }
        }
    }

    // Adjust the pipeline and partitioning approach
    // Use a buffer for input access to reduce memory port pressure
    load_input: for (int n = 0; n < n_limit; n++) {
        for (int h = 0; h < INPUT_TILE_HEIGHT; h++) {
            // Load a row of data into a local buffer first
            data_t row_buffer[INPUT_TILE_WIDTH];
            #pragma HLS ARRAY_PARTITION variable=row_buffer cyclic factor=2
            
            load_row: for (int w = 0; w < INPUT_TILE_WIDTH; w++) {
                #pragma HLS PIPELINE II=2  // Relaxed II to reduce port pressure
                // Get position in original input (considering padding)
                int input_h = h + h_offset*S - P;
                int input_w = w + w_offset*S - P;
                
                // Only load data that is within bounds
                if (input_h >= 0 && input_h < H && input_w >= 0 && input_w < W) {
                    int input_idx = (n + n_offset) * H * W + input_h * W + input_w;
                    row_buffer[w] = input_ddr[input_idx];
                } else {
                    row_buffer[w] = 0;
                }
            }
            
            // Now transfer from row buffer to input buffer
            transfer_row: for (int w = 0; w < INPUT_TILE_WIDTH; w++) {
                #pragma HLS PIPELINE II=1
                input_buffer[n][h][w] = row_buffer[w];
            }
        }
    }
}

// Function to load weights from DDR to on-chip buffer
void load_weight_tile(
    data_t* weights_ddr,
    data_t weight_buffer[TM][TN][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE],
    int m_offset, int n_offset,
    int M, int N, int K) {
    
    #pragma HLS INLINE off
    
    // Pre-compute limits
    const int m_limit = ((m_offset + TM) > M) ? (M - m_offset) : TM;
    const int n_limit = ((n_offset + TN) > N) ? (N - n_offset) : TN;
    const int K2 = K*K;
    
    // Use a more structured approach with fixed-size inner loops
    // Initialize all weights to zero
    clear_weights: for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            #pragma HLS PIPELINE II=1
            for (int k = 0; k < MAX_KERNEL_SIZE*MAX_KERNEL_SIZE; k++) {
                weight_buffer[m][n][k] = 0;
            }
        }
    }
    
    // Load weights with a more HLS-friendly approach
    load_weights_m: for (int m = 0; m < m_limit; m++) {
        load_weights_n: for (int n = 0; n < n_limit; n++) {
            // Use a smaller batch size with fixed iterations
            load_weights_k: for (int k_base = 0; k_base < K2; k_base += 4) {
                // Calculate the actual batch size (handle edge case)
                int k_limit = ((k_base + 4) <= K2) ? 4 : (K2 - k_base);
                
                // Load a batch of kernel weights
                #pragma HLS PIPELINE II=1
                load_weights_k_inner: for (int k_offset = 0; k_offset < k_limit; k_offset++) {
                    int k = k_base + k_offset;
                    int weight_idx = (m + m_offset) * N * K2 + (n + n_offset) * K2 + k;
                    weight_buffer[m][n][k] = weights_ddr[weight_idx];
                }
            }
        }
    }
}

// Function to load bias values from DDR to on-chip buffer
void load_bias(
    data_t* bias_ddr,
    data_t bias_buffer[TM],
    int m_offset, int M) {
    
    #pragma HLS INLINE off
    
    // Pre-compute limit
    const int m_limit = ((m_offset + TM) > M) ? (M - m_offset) : TM;
    
    // Clear all bias values first
    clear_bias: for (int m = 0; m < TM; m++) {
        #pragma HLS PIPELINE II=1
        bias_buffer[m] = 0;
    }
    
    // Then load actual bias values with optimized pipeline
    load_bias: for (int m = 0; m < m_limit; m++) {
        #pragma HLS PIPELINE II=1
        bias_buffer[m] = bias_ddr[m + m_offset];
    }
}

// Function to store output feature map from on-chip buffer to DDR
void store_output_tile(
    data_t* output_ddr,
    data_t output_buffer[TM][TR][TC],
    int m_offset, int h_offset, int w_offset,
    int M, int R, int C) {
    
    #pragma HLS INLINE off
    
    // Pre-compute limits
    const int m_limit = ((m_offset + TM) > M) ? (M - m_offset) : TM;
    const int r_limit = ((h_offset + TR) > R) ? (R - h_offset) : TR;
    const int c_limit = ((w_offset + TC) > C) ? (C - w_offset) : TC;

    // Use a buffer-based approach to help with optimization
    store_output: for (int m = 0; m < m_limit; m++) {
        for (int r = 0; r < r_limit; r++) {
            // Prepare row buffer for burst write
            data_t row_buffer[TC];
            #pragma HLS ARRAY_PARTITION variable=row_buffer cyclic factor=2
            
            // Fill row buffer
            fill_row: for (int c = 0; c < c_limit; c++) {
                #pragma HLS PIPELINE II=1
                row_buffer[c] = output_buffer[m][r][c];
            }
            
            // Store row buffer to memory with relaxed II
            store_row: for (int c = 0; c < c_limit; c++) {
                #pragma HLS PIPELINE II=2
                int output_idx = (m + m_offset) * R * C + (r + h_offset) * C + (c + w_offset);
                output_ddr[output_idx] = row_buffer[c];
            }
        }
    }
}