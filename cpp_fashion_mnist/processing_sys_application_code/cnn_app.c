/**
 * Fashion MNIST CNN Accelerator - Full Debug Version
 * For AMD Kria KV260 Vision AI Starter Kit
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "xil_printf.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xparameters.h"
#include "sleep.h"
#include "cnn_params.h"
#include "cnn_data.h"

// Accelerator control interfaces
#define ACCEL_CTRL_BASEADDR    0xA0000000  // s_axi_control
#define ACCEL_CTRL_R_BASEADDR  0xA0010000  // s_axi_control_r

// Control register bits
#define AP_CTRL_START_BIT      0x1
#define AP_CTRL_DONE_BIT       0x2
#define AP_CTRL_IDLE_BIT       0x4

#define CTRL_REG_OFFSET        0x00

// DDR memory regions - using AXI interface base addresses from updated address map
#define IMAGE_DDR_ADDR         0x04000000UL  // gmem0: 0x4000000 (64MB)
#define CONV1_WEIGHTS_DDR_ADDR 0x08000000UL  // gmem1: 0x8000000 (128MB)
#define CONV1_BIAS_DDR_ADDR    0x10000000UL  // gmem2: 0x10000000 (128MB)
#define CONV2_WEIGHTS_DDR_ADDR 0x18000000UL  // gmem3: 0x18000000 (128MB)
#define CONV2_BIAS_DDR_ADDR    0x20000000UL  // gmem4: 0x20000000 (128MB)
#define FC1_WEIGHTS_DDR_ADDR   0x28000000UL  // gmem5: 0x28000000 (128MB)
#define FC1_BIAS_DDR_ADDR      0x30000000UL  // gmem6: 0x30000000 (128MB)
#define FC2_WEIGHTS_DDR_ADDR   0x38000000UL  // gmem7: 0x38000000 (128MB)
#define FC2_BIAS_DDR_ADDR      0x40000000UL  // gmem8: 0x40000000 (128MB)
#define PREDICTIONS_DDR_ADDR   0x48000000UL  // gmem9: 0x48000000 (128MB)

#define CNN_SUCCESS            0
#define CNN_ERROR_ACCELERATOR  2
#define CNN_ERROR_DATA         3
#define CNN_ERROR_TIMEOUT      4

// ap_fixed<16,5> conversion
typedef struct {
    short data;
} fixed16_5_t;

fixed16_5_t float_to_fixed16_5(float f) {
    fixed16_5_t result;
    if (f > 15.999f) f = 15.999f;
    if (f < -16.0f) f = -16.0f;
    result.data = (short)(f * (1 << 11));
    return result;
}

float fixed16_5_to_float(fixed16_5_t fixed) {
    return (float)fixed.data / (1 << 11);
}

void print_float(float f) {
    int whole = (int)f;
    int frac = (int)((f - whole) * 1000);
    if (frac < 0) frac = -frac;
    xil_printf("%d.%03d", whole, frac);
}

void copy_float_to_fixed_ddr(uintptr_t ddr_addr, const float* src_data, size_t count, const char* name) {
    fixed16_5_t* dest = (fixed16_5_t*)ddr_addr;

    xil_printf("  Copying %s (%d values) to 0x%08x:\r\n", name, (int)count, (u32)ddr_addr);

    for (size_t i = 0; i < count; i++) {
        dest[i] = float_to_fixed16_5(src_data[i]);
    }

    Xil_DCacheFlushRange((UINTPTR)ddr_addr, count * sizeof(fixed16_5_t));

    // Verify first few values
    xil_printf("    First 3 values: ");
    for (int i = 0; i < 3 && i < count; i++) {
        print_float(fixed16_5_to_float(dest[i]));
        xil_printf(" ");
    }
    xil_printf("\r\n");
}

int reset_accelerator(void) {
    u32 status = Xil_In32(ACCEL_CTRL_BASEADDR + CTRL_REG_OFFSET);
    xil_printf("  Initial status: 0x%08x\r\n", status);

    if (status & AP_CTRL_IDLE_BIT) {
        xil_printf("  Already IDLE\r\n");
        return CNN_SUCCESS;
    }

    Xil_Out32(ACCEL_CTRL_BASEADDR + CTRL_REG_OFFSET, 0);
    usleep(1000);

    status = Xil_In32(ACCEL_CTRL_BASEADDR + CTRL_REG_OFFSET);
    xil_printf("  After reset: 0x%08x\r\n", status);

    if (!(status & AP_CTRL_IDLE_BIT)) {
        xil_printf("  ERROR: Reset failed\r\n");
        return CNN_ERROR_ACCELERATOR;
    }
    return CNN_SUCCESS;
}

int wait_for_accelerator_completion(int timeout_ms) {
    u32 status;
    int elapsed_ms = 0;

    xil_printf("  Starting accelerator, waiting for completion...\r\n");

    do {
        status = Xil_In32(ACCEL_CTRL_BASEADDR + CTRL_REG_OFFSET);

        if (status & AP_CTRL_DONE_BIT) {
            xil_printf("  Accelerator completed, final status: 0x%08x\r\n", status);
            return CNN_SUCCESS;
        }

        if (elapsed_ms % 200 == 0) {
            xil_printf("    %d ms elapsed, status: 0x%08x\r\n", elapsed_ms, status);
        }

        usleep(10000);
        elapsed_ms += 10;
    } while (elapsed_ms < timeout_ms);
    
    xil_printf("  ERROR: Timeout after %d ms, status: 0x%08x\r\n", timeout_ms, status);
    return CNN_ERROR_TIMEOUT;
}

void scan_ddr_for_predictions(void) {
    xil_printf("  Scanning DDR for actual prediction writes...\r\n");

    // Check a few key locations where predictions might be
    u32 check_addrs[] = {
        0x04000000, 0x08000000, 0x10000000, 0x18000000, 0x20000000,
        0x28000000, 0x30000000, 0x38000000, 0x40000000, 0x48000000,
        0x50000000, 0x60000000
    };

    for (int addr_idx = 0; addr_idx < 12; addr_idx++) {
        fixed16_5_t* check_ptr = (fixed16_5_t*)check_addrs[addr_idx];

        // Look for 10 consecutive non-(-16.0) values
        bool found_data = false;
        for (int offset = 0; offset < 1000; offset += 10) {
            int valid_count = 0;
            for (int i = 0; i < 10; i++) {
                float val = fixed16_5_to_float(check_ptr[offset + i]);
                if (val != -16.0f && val != 0.0f && val > -15.0f && val < 15.0f) {
                    valid_count++;
                }
            }

            if (valid_count >= 8) {  // At least 8 out of 10 look like valid predictions
                xil_printf("    FOUND POTENTIAL PREDICTIONS at 0x%08x + %d:\r\n",
                          check_addrs[addr_idx], offset * 2);
                for (int i = 0; i < 10; i++) {
                    xil_printf("      [%d]: ", i);
                    print_float(fixed16_5_to_float(check_ptr[offset + i]));
                    xil_printf("\r\n");
                }
                found_data = true;
                break;
            }
        }

        if (!found_data) {
            // Check first few values for any non-default data
            xil_printf("    0x%08x: ", check_addrs[addr_idx]);
            for (int i = 0; i < 5; i++) {
                print_float(fixed16_5_to_float(check_ptr[i]));
                xil_printf(" ");
            }
            xil_printf("\r\n");
        }
    }
}

int run_cnn_single_image(void) {
    int sample_idx = 0;

    xil_printf("\r\n=== Running CNN inference with full debugging ===\r\n");

    // Step 1: Convert and copy data to DDR with verification
    xil_printf("\r\nStep 1: Converting and copying data to DDR...\r\n");

    copy_float_to_fixed_ddr(IMAGE_DDR_ADDR, test_samples[sample_idx].image_data,
                           IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE, "image");
    copy_float_to_fixed_ddr(CONV1_WEIGHTS_DDR_ADDR, conv1_weights,
                           CONV1_FILTERS * CONV1_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE, "conv1_weights");
    copy_float_to_fixed_ddr(CONV1_BIAS_DDR_ADDR, conv1_bias, CONV1_FILTERS, "conv1_bias");
    copy_float_to_fixed_ddr(CONV2_WEIGHTS_DDR_ADDR, conv2_weights,
                           CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE, "conv2_weights");
    copy_float_to_fixed_ddr(CONV2_BIAS_DDR_ADDR, conv2_bias, CONV2_FILTERS, "conv2_bias");
    copy_float_to_fixed_ddr(FC1_WEIGHTS_DDR_ADDR, fc1_weights, FC1_WEIGHTS_H * FC1_WEIGHTS_W, "fc1_weights");
    copy_float_to_fixed_ddr(FC1_BIAS_DDR_ADDR, fc1_bias, FC1_WEIGHTS_W, "fc1_bias");
    copy_float_to_fixed_ddr(FC2_WEIGHTS_DDR_ADDR, fc2_weights, FC1_WEIGHTS_W * FC2_WEIGHTS_W, "fc2_weights");
    copy_float_to_fixed_ddr(FC2_BIAS_DDR_ADDR, fc2_bias, FC2_WEIGHTS_W, "fc2_bias");

    // Initialize predictions buffer with marker values
    xil_printf("  Initializing predictions buffer at 0x%08x\r\n", (u32)PREDICTIONS_DDR_ADDR);
    fixed16_5_t* predictions_ptr = (fixed16_5_t*)PREDICTIONS_DDR_ADDR;
    for (int i = 0; i < FC2_WEIGHTS_W; i++) {
        predictions_ptr[i] = float_to_fixed16_5(-999.0f);
    }
    Xil_DCacheFlushRange(PREDICTIONS_DDR_ADDR, FC2_WEIGHTS_W * sizeof(fixed16_5_t));

    // Step 2: Reset accelerator
    xil_printf("\r\nStep 2: Reset accelerator...\r\n");
    if (reset_accelerator() != CNN_SUCCESS) {
        return CNN_ERROR_ACCELERATOR;
    }

    // Step 3: Configure accelerator addresses with full verification
    xil_printf("\r\nStep 3: Configure accelerator addresses...\r\n");

    u32 ctrl_base = ACCEL_CTRL_R_BASEADDR;
    xil_printf("  Using s_axi_control_r at 0x%08x\r\n", ctrl_base);

    struct {
        int offset;
        u32 addr;
        const char* name;
    } addr_regs[] = {
        {0x10, (u32)IMAGE_DDR_ADDR, "image"},
        {0x20, (u32)CONV1_WEIGHTS_DDR_ADDR, "conv1_weights"},
        {0x28, (u32)CONV1_BIAS_DDR_ADDR, "conv1_bias"},
        {0x38, (u32)CONV2_WEIGHTS_DDR_ADDR, "conv2_weights"},
        {0x40, (u32)CONV2_BIAS_DDR_ADDR, "conv2_bias"},
        {0x50, (u32)FC1_WEIGHTS_DDR_ADDR, "fc1_weights"},
        {0x58, (u32)FC1_BIAS_DDR_ADDR, "fc1_bias"},
        {0x68, (u32)FC2_WEIGHTS_DDR_ADDR, "fc2_weights"},
        {0x70, (u32)FC2_BIAS_DDR_ADDR, "fc2_bias"},
        {0x80, (u32)PREDICTIONS_DDR_ADDR, "predictions"}
    };

    bool config_success = true;
    for (int i = 0; i < 10; i++) {
        xil_printf("  Setting %s: 0x%08x -> offset 0x%02x\r\n",
                  addr_regs[i].name, addr_regs[i].addr, addr_regs[i].offset);

        Xil_Out32(ctrl_base + addr_regs[i].offset, addr_regs[i].addr);
        u32 readback = Xil_In32(ctrl_base + addr_regs[i].offset);

        if (readback == addr_regs[i].addr) {
            xil_printf("    ✓ Verified: 0x%08x\r\n", readback);
        } else {
            xil_printf("    ❌ FAILED: wrote 0x%08x, read 0x%08x\r\n",
                      addr_regs[i].addr, readback);
            config_success = false;
        }
    }

    if (!config_success) {
        xil_printf("  ERROR: Address configuration failed\r\n");
        return CNN_ERROR_ACCELERATOR;
    }

    // Step 4: Start accelerator
    xil_printf("\r\nStep 4: Start accelerator...\r\n");
    xil_printf("  Writing START bit to control register at 0x%08x\r\n",
              ACCEL_CTRL_BASEADDR + CTRL_REG_OFFSET);

    Xil_Out32(ACCEL_CTRL_BASEADDR + CTRL_REG_OFFSET, AP_CTRL_START_BIT);

    int result = wait_for_accelerator_completion(15000);
    if (result != CNN_SUCCESS) {
        xil_printf("  Accelerator failed to complete\r\n");
        return result;
    }

    // Step 5: Check predictions buffer first
    xil_printf("\r\nStep 5: Checking prediction results...\r\n");
    Xil_DCacheInvalidateRange(PREDICTIONS_DDR_ADDR, FC2_WEIGHTS_W * sizeof(fixed16_5_t));

    xil_printf("  Predictions at 0x%08x:\r\n", (u32)PREDICTIONS_DDR_ADDR);
    bool data_changed = false;
    for (int i = 0; i < FC2_WEIGHTS_W; i++) {
        float pred_val = fixed16_5_to_float(predictions_ptr[i]);
        xil_printf("    pred[%d]: ", i);
        print_float(pred_val);
        xil_printf(" (0x%04x)\r\n", predictions_ptr[i].data);

        if (pred_val != -999.0f) {
            data_changed = true;
        }
    }

    if (!data_changed) {
        xil_printf("\r\n  ERROR: Predictions buffer unchanged!\r\n");
        xil_printf("  Accelerator completed but didn't write to prediction address.\r\n");

        // Scan DDR to find where it actually wrote
        scan_ddr_for_predictions();

        return CNN_ERROR_DATA;
    } else {
        xil_printf("\r\n  Predictions buffer was modified, but all values are -16.0\r\n");
        xil_printf("  This suggests numerical saturation in HLS computation.\r\n");

        // Still scan to see if there are better predictions elsewhere
        scan_ddr_for_predictions();
    }

    // Find predicted class
    float predictions[FC2_WEIGHTS_W];
    for (int i = 0; i < FC2_WEIGHTS_W; i++) {
        predictions[i] = fixed16_5_to_float(predictions_ptr[i]);
    }

    int predicted_class = 0;
    float max_prob = predictions[0];
    for (int i = 1; i < FC2_WEIGHTS_W; i++) {
        if (predictions[i] > max_prob) {
            max_prob = predictions[i];
            predicted_class = i;
        }
    }

    xil_printf("\r\n  SUCCESS! Predicted class: %d (%s), confidence: ",
               predicted_class, FASHION_CLASSES[predicted_class]);
    print_float(max_prob);
    xil_printf("\r\n");

    return CNN_SUCCESS;
}

int main() {
    xil_printf("\r\n=============================================================\r\n");
    xil_printf("CNN Accelerator - Full Debug Version\r\n");
    xil_printf("=============================================================\r\n");

    int status = run_cnn_single_image();

    if (status == CNN_SUCCESS) {
        xil_printf("\r\nTest PASSED!\r\n");
    } else {
        xil_printf("\r\nTest FAILED with error %d\r\n", status);
    }

    return status;
}
