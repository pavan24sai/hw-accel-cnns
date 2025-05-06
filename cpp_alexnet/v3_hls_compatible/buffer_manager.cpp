#include "cnn_functions.h"

// Function to initialize output buffer with bias values
void init_output_buffer(
    data_t output_buffer[TM][TR][TC], 
    data_t bias_buffer[TM],
    int tm_bound) {
    
    #pragma HLS INLINE
    
    // Simple loop structure with straightforward pipelining
    init_m: for (int too = 0; too < TM; too++) {
        init_r: for (int trr = 0; trr < TR; trr++) {
            #pragma HLS PIPELINE II=1
            init_c: for (int tcc = 0; tcc < TC; tcc++) {
                // Initialize with bias if within valid bounds, otherwise zero
                output_buffer[too][trr][tcc] = (too < tm_bound) ? bias_buffer[too] : data_t(0);
            }
        }
    }
}