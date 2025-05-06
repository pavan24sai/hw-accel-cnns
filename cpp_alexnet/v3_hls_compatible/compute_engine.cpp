#include "cnn_functions.h"

// The core computation engine implementing the optimized loop ordering from the paper
void compute_tile(
    data_t input_buffer[TN][INPUT_TILE_HEIGHT][INPUT_TILE_WIDTH],
    data_t weight_buffer[TM][TN][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE],
    data_t output_buffer[TM][TR][TC],
    int kernel_size, int stride, int tm_bound, int tn_bound, int tr_bound, int tc_bound) {
    
    #pragma HLS INLINE off
    
    // Pre-compute limits to help HLS optimization
    const int k2 = kernel_size * kernel_size;
    
    // Loop ordered as in the paper: i -> j -> trr -> tcc -> too -> tii
    i_loop: for (int i = 0; i < kernel_size; i++) {
        j_loop: for (int j = 0; j < kernel_size; j++) {
            trr_loop: for (int trr = 0; trr < tr_bound; trr++) {
                tcc_loop: for (int tcc = 0; tcc < tc_bound; tcc++) {
                    // Calculate input position
                    int h = trr * stride + i;
                    int w = tcc * stride + j;
                    int k_idx = i * kernel_size + j;
                    
                    // Process in smaller batches while preserving the Zhang paper's core loop ordering pattern
					// Adding lot of pragmas in this loop (like high unrolling and tiling factors) slows down the HLS process significantly.
					// So, tried with smaller batch sizes & unrolling the inner loops.
                    too_batch_loop: for (int too_base = 0; too_base < tm_bound; too_base += 2) {
                        // Calculate actual batch size (handle edge case)
                        int too_limit = ((too_base + 2) <= tm_bound) ? 2 : (tm_bound - too_base);
                        
                        // Main computation with optimized pragmas
                        #pragma HLS PIPELINE II=1
                        tii_loop: for (int tii = 0; tii < tn_bound; tii++) {
                            // Process each output feature map in this batch
                            too_inner_loop: for (int too_offset = 0; too_offset < too_limit; too_offset++) {
                                #pragma HLS UNROLL
                                int too = too_base + too_offset;
                                
                                // Main convolution operation
                                output_buffer[too][trr][tcc] += 
                                    weight_buffer[too][tii][k_idx] * 
                                    input_buffer[tii][h][w];
                            }
                        }
                    }
                }
            }
        }
    }
}

// Function to apply ReLU activation
void apply_relu(data_t buffer[TM][TR][TC], int tm, int tr, int tc) {
    #pragma HLS INLINE off
    
    // Simple loop structure for ReLU
    m_loop: for (int too = 0; too < tm; too++) {
        r_loop: for (int trr = 0; trr < tr; trr++) {
            #pragma HLS PIPELINE II=1
            c_loop: for (int tcc = 0; tcc < tc; tcc++) {
                // ReLU function: max(0, x)
                if (buffer[too][trr][tcc] < 0) {
                    buffer[too][trr][tcc] = 0;
                }
            }
        }
    }
}