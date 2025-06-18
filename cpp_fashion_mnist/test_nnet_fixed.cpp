/******************************************************************************
 * Test Bench with EXTENSIVE DEBUG for C/RTL CO-SIMULATION
 * - Improved data loading and memory handling for C/RTL co-simulation
 * - Fixed array boundary checks and indexing
 ******************************************************************************/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <sstream>
#include "headers/defines.h"
#include "headers/activations.h"

// Function declaration for the nnet accelerator
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
);

// Fashion-MNIST class names
const std::vector<std::string> FASHION_CLASSES = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};

// Global weight arrays with correct sizes
float24_t g_conv1_weights[CONV1_FILTERS * CONV1_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE];
float24_t g_conv1_bias[CONV1_FILTERS];
float24_t g_conv2_weights[CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE];
float24_t g_conv2_bias[CONV2_FILTERS];
float24_t g_fc1_weights[FC1_WEIGHTS_H * FC1_WEIGHTS_W];
float24_t g_fc1_bias[FC1_WEIGHTS_W];
float24_t g_fc2_weights[FC1_WEIGHTS_W * FC2_WEIGHTS_W];
float24_t g_fc2_bias[FC2_WEIGHTS_W];

// Global arrays for input image and predictions
float24_t g_image[IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE];
float24_t g_predictions[FC2_WEIGHTS_W];

// DEBUG HELPER FUNCTIONS
void print_debug_array(const std::string& name, const float24_t* arr, int size, int max_print = 10) {
    std::cout << "[TESTBENCH DEBUG] " << name << " (first " << max_print << " of " << size << "): ";
    for (int i = 0; i < std::min(max_print, size); i++) {
        std::cout << std::fixed << std::setprecision(6) << arr[i] << " ";
    }
    std::cout << std::endl;
}

void verify_array_non_zero(const std::string& name, const float24_t* arr, int size) {
    int non_zero_count = 0;
    float24_t sum = 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] != float24_t(0)) {
            non_zero_count++;
            sum += arr[i];
        }
    }
    std::cout << "[TESTBENCH DEBUG] " << name << " - Non-zero elements: " << non_zero_count 
              << "/" << size << ", Sum: " << sum << std::endl;
}

bool load_weights_from_file(const std::string& filename, float24_t* weights, int expected_count) {
    std::cout << "[TESTBENCH DEBUG] Loading weights from: " << filename << std::endl;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open weight file: " << filename << std::endl;
        return false;
    }
    
    // Initialize to zero first
    for (int i = 0; i < expected_count; i++) {
        weights[i] = float24_t(0);
    }
    
    // Read weights as 32-bit floats
    for (int i = 0; i < expected_count; i++) {
        float weight;
        file.read(reinterpret_cast<char*>(&weight), sizeof(float));
        if (file.fail()) {
            std::cerr << "Error: Failed to read weight " << i << " from " << filename << std::endl;
            file.close();
            return false;
        }
        weights[i] = static_cast<float24_t>(weight);
    }
    
    file.close();
    std::cout << "Successfully loaded " << expected_count << " weights from " << filename << std::endl;
    
    // DEBUG: Print first few weights and verify non-zero
    print_debug_array("Loaded " + filename, weights, expected_count, 10);
    verify_array_non_zero("Loaded " + filename, weights, expected_count);
    
    return true;
}

bool load_all_weights() {
    std::cout << "============ LOADING ALL WEIGHTS WITH DEBUG ============" << std::endl;
    
    const std::string weight_dir = "/home/pavan/WorkArea/UW/EE470_CAII/cursor_modules/ver8/weights/";
    
    // Load Conv1 weights and bias
    if (!load_weights_from_file(weight_dir + "conv1_weights.bin", g_conv1_weights, 
                               CONV1_FILTERS * CONV1_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE)) {
        return false;
    }
    if (!load_weights_from_file(weight_dir + "conv1_bias.bin", g_conv1_bias, CONV1_FILTERS)) {
        return false;
    }
    
    // Load Conv2 weights and bias
    if (!load_weights_from_file(weight_dir + "conv2_weights.bin", g_conv2_weights,
                               CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE)) {
        return false;
    }
    if (!load_weights_from_file(weight_dir + "conv2_bias.bin", g_conv2_bias, CONV2_FILTERS)) {
        return false;
    }
    
    // Load FC1 weights and bias
    if (!load_weights_from_file(weight_dir + "fc1_weights.bin", g_fc1_weights,
                               FC1_WEIGHTS_H * FC1_WEIGHTS_W)) {
        return false;
    }
    if (!load_weights_from_file(weight_dir + "fc1_bias.bin", g_fc1_bias, FC1_WEIGHTS_W)) {
        return false;
    }
    
    // Load FC2 weights and bias
    if (!load_weights_from_file(weight_dir + "fc2_weights.bin", g_fc2_weights,
                               FC1_WEIGHTS_W * FC2_WEIGHTS_W)) {
        return false;
    }
    if (!load_weights_from_file(weight_dir + "fc2_bias.bin", g_fc2_bias, FC2_WEIGHTS_W)) {
        return false;
    }
    
    std::cout << "✓ All weights loaded successfully!" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    return true;
}

bool load_test_image(const std::string& filename, float24_t* image) {
    std::cout << "[TESTBENCH DEBUG] Loading test image from: " << filename << std::endl;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open test image file: " << filename << std::endl;
        return false;
    }
    
    // Initialize to zero first
    for (int i = 0; i < IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE; i++) {
        image[i] = float24_t(0);
    }
    
    // Read image as 32-bit floats
    for (int i = 0; i < IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE; i++) {
        float pixel;
        file.read(reinterpret_cast<char*>(&pixel), sizeof(float));
        if (file.fail()) {
            std::cerr << "Error: Failed to read pixel " << i << " from " << filename << std::endl;
            file.close();
            return false;
        }
        image[i] = static_cast<float24_t>(pixel);
    }
    
    file.close();
    std::cout << "Successfully loaded test image from file: " << filename << std::endl;
    
    // DEBUG: Print first few pixels and verify non-zero
    print_debug_array("Loaded image", image, IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE, 10);
    verify_array_non_zero("Loaded image", image, IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE);
    
    return true;
}

void print_image_sample(const float24_t* image, int sample_size = 8) {
    std::cout << "Image sample (first " << sample_size << "x" << sample_size << " pixels):" << std::endl;
    for (int i = 0; i < sample_size; i++) {
        for (int j = 0; j < sample_size; j++) {
            int idx = i * IMAGE_SIZE + j;
            std::cout << std::fixed << std::setprecision(2) << std::setw(5) << image[idx] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void print_predictions(const float24_t* predictions) {
    std::cout << "Prediction probabilities:" << std::endl;
    
    // Find the class with highest probability
    int predicted_class = 0;
    float24_t max_prob = predictions[0];
    
    for (int i = 0; i < FC2_WEIGHTS_W; i++) {
        if (predictions[i] > max_prob) {
            max_prob = predictions[i];
            predicted_class = i;
        }
        std::cout << "  " << i << " (" << FASHION_CLASSES[i] << "): " 
                  << std::fixed << std::setprecision(6) << predictions[i] << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Predicted class: " << predicted_class << " (" << FASHION_CLASSES[predicted_class] 
              << ") with confidence: " << std::fixed << std::setprecision(6) << max_prob << std::endl;
}

struct TestInfo {
    std::string filename;
    int expected_class;
    std::string class_name;
};

std::vector<TestInfo> load_test_info(const std::string& info_file) {
    std::vector<TestInfo> test_info;
    std::ifstream file(info_file);
    
    if (!file.is_open()) {
        std::cerr << "Warning: Cannot open test info file: " << info_file << std::endl;
        return test_info;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string index_str, filename, label_str, class_name;
        
        if (std::getline(iss, index_str, ',') &&
            std::getline(iss, filename, ',') &&
            std::getline(iss, label_str, ',') &&
            std::getline(iss, class_name)) {
            
            int expected_class = std::stoi(label_str);
            test_info.push_back({filename, expected_class, class_name});
        }
    }
    
    file.close();
    std::cout << "Loaded " << test_info.size() << " test samples from " << info_file << std::endl;
    return test_info;
}

void run_batch_test(int num_samples = 10) {
    std::cout << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Running batch test on " << num_samples << " samples..." << std::endl;
    std::cout << "============================================================" << std::endl;
    
    const std::string test_info_path = "/home/pavan/WorkArea/UW/EE470_CAII/cursor_modules/ver8/test_dataset/test_info.txt";
    const std::string test_dataset_dir = "/home/pavan/WorkArea/UW/EE470_CAII/cursor_modules/ver8/test_dataset/";
    
    std::vector<TestInfo> test_info = load_test_info(test_info_path);
    
    if (test_info.empty()) {
        std::cerr << "Error: No test samples found. Please run generate_test_dataset.py first." << std::endl;
        return;
    }
    
    int actual_samples = std::min(num_samples, static_cast<int>(test_info.size()));
    int correct_predictions = 0;
    
    std::cout << "Testing " << actual_samples << " samples from " << test_info.size() << " available..." << std::endl;
    
    // Start timing for the entire process
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Measure weights loading time separately
    auto weights_load_start = std::chrono::high_resolution_clock::now();
    
    // Weight loading is already being done in the main function before calling this function
    // If you want to measure it here, you would need to reload the weights
    
    auto weights_load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> weights_load_time = weights_load_end - weights_load_start;
    
    // Batch processing with timing
    auto batch_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < actual_samples; i++) {
        const TestInfo& info = test_info[i];
        std::string image_path = test_dataset_dir + info.filename;
        
        std::cout << "\n============ SAMPLE " << (i + 1) << " DEBUG ============" << std::endl;
        
        // Clear predictions array and initialize to recognizable values
        for (int j = 0; j < FC2_WEIGHTS_W; j++) {
            g_predictions[j] = float24_t(-999.0f);  // Use recognizable initial value
        }
        
        std::cout << "[TESTBENCH DEBUG] Cleared predictions array" << std::endl;
        print_debug_array("Initial predictions", g_predictions, FC2_WEIGHTS_W, 10);
        
        // Measure image loading time
        auto image_load_start = std::chrono::high_resolution_clock::now();
        
        // Load test image
        if (!load_test_image(image_path, g_image)) {
            std::cerr << "Failed to load test image: " << image_path << std::endl;
            continue;
        }
        
        auto image_load_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> image_load_time = image_load_end - image_load_start;
        
        std::cout << "[TESTBENCH DEBUG] ===== PRE-INFERENCE DATA VERIFICATION =====" << std::endl;
        print_debug_array("PRE-INFERENCE image", g_image, IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE, 10);
        print_debug_array("PRE-INFERENCE conv1_weights", g_conv1_weights, 10, 10);
        print_debug_array("PRE-INFERENCE conv1_bias", g_conv1_bias, CONV1_FILTERS, 32);
        print_debug_array("PRE-INFERENCE conv2_bias", g_conv2_bias, CONV2_FILTERS, 64);
        print_debug_array("PRE-INFERENCE fc1_bias", g_fc1_bias, FC1_WEIGHTS_W, 10);
        print_debug_array("PRE-INFERENCE fc2_bias", g_fc2_bias, FC2_WEIGHTS_W, 10);
        
        std::cout << "[TESTBENCH DEBUG] About to call nnet() function..." << std::endl;
        
        // Measure inference time
        auto inference_start = std::chrono::high_resolution_clock::now();
        
        // Run inference with explicit array passing
        nnet(g_image, g_conv1_weights, g_conv1_bias, g_conv2_weights, g_conv2_bias,
             g_fc1_weights, g_fc1_bias, g_fc2_weights, g_fc2_bias, g_predictions);
        
        auto inference_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = inference_end - inference_start;
        
        std::cout << "[TESTBENCH DEBUG] ===== POST-INFERENCE DATA VERIFICATION =====" << std::endl;
        print_debug_array("POST-INFERENCE predictions", g_predictions, FC2_WEIGHTS_W, 10);
        verify_array_non_zero("POST-INFERENCE predictions", g_predictions, FC2_WEIGHTS_W);
        
        // Check if all predictions are the same (indication of a problem)
        bool all_same = true;
        for (int j = 1; j < FC2_WEIGHTS_W; j++) {
            if (g_predictions[j] != g_predictions[0]) {
                all_same = false;
                break;
            }
        }
        if (all_same) {
            std::cout << "[TESTBENCH DEBUG] WARNING: All predictions are the same value: " << g_predictions[0] << std::endl;
        }
        
        // Find predicted class
        int predicted_class = 0;
        float24_t max_prob = g_predictions[0];
        for (int j = 1; j < FC2_WEIGHTS_W; j++) {
            if (g_predictions[j] > max_prob) {
                max_prob = g_predictions[j];
                predicted_class = j;
            }
        }
        
        std::cout << "[TESTBENCH DEBUG] Found max probability: " << max_prob << " at class: " << predicted_class << std::endl;
        
        // Check if prediction is correct
        bool is_correct = (predicted_class == info.expected_class);
        if (is_correct) {
            correct_predictions++;
        }
        
        std::cout << "Sample " << std::setw(3) << (i + 1) << "/" << actual_samples << ": " 
                  << "Expected: " << info.expected_class << " (" << FASHION_CLASSES[info.expected_class] << "), "
                  << "Predicted: " << predicted_class << " (" << FASHION_CLASSES[predicted_class] << "), "
                  << "Confidence: " << std::fixed << std::setprecision(6) << max_prob << " "
                  << (is_correct ? "✓" : "✗") << std::endl;
        
        // Print timing for this sample
        std::cout << "Image load time: " << image_load_time.count() << " ms" << std::endl;
        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;
        
        std::cout << "============ END SAMPLE " << (i + 1) << " DEBUG ============\n" << std::endl;
    }
    
    auto batch_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> batch_time = batch_end - batch_start;
    
    // End timing for the entire process
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = end_time - start_time;
    
    // Calculate and display accuracy
    float accuracy = static_cast<float>(correct_predictions) / actual_samples * 100.0f;
    
    std::cout << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Batch Test Results:" << std::endl;
    std::cout << "  Total samples tested: " << actual_samples << std::endl;
    std::cout << "  Correct predictions: " << correct_predictions << std::endl;
    std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Timing Information:" << std::endl;
    std::cout << "  Weights load time: " << weights_load_time.count() << " ms" << std::endl;
    std::cout << "  Batch processing time: " << batch_time.count() << " ms" << std::endl;
    std::cout << "  Average inference time per sample: " << (batch_time.count() / actual_samples) << " ms" << std::endl;
    std::cout << "  Total execution time: " << total_time.count() << " ms" << std::endl;
    std::cout << "============================================================" << std::endl;
}

int main() {
    std::cout << "Fashion-MNIST CNN Accelerator Test Bench (EXTENSIVE DEBUG FOR C/RTL CO-SIMULATION)" << std::endl;
    std::cout << "===================================================================================" << std::endl;

    // Start timing the entire process
    auto total_start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize all arrays to recognizable values
    for (int i = 0; i < CONV1_FILTERS * CONV1_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE; i++) {
        g_conv1_weights[i] = float24_t(-888.0f);
    }
    for (int i = 0; i < CONV1_FILTERS; i++) {
        g_conv1_bias[i] = float24_t(-888.0f);
    }
    for (int i = 0; i < CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE; i++) {
        g_conv2_weights[i] = float24_t(-888.0f);
    }
    for (int i = 0; i < CONV2_FILTERS; i++) {
        g_conv2_bias[i] = float24_t(-888.0f);
    }
    for (int i = 0; i < FC1_WEIGHTS_H * FC1_WEIGHTS_W; i++) {
        g_fc1_weights[i] = float24_t(-888.0f);
    }
    for (int i = 0; i < FC1_WEIGHTS_W; i++) {
        g_fc1_bias[i] = float24_t(-888.0f);
    }
    for (int i = 0; i < FC1_WEIGHTS_W * FC2_WEIGHTS_W; i++) {
        g_fc2_weights[i] = float24_t(-888.0f);
    }
    for (int i = 0; i < FC2_WEIGHTS_W; i++) {
        g_fc2_bias[i] = float24_t(-888.0f);
    }
    for (int i = 0; i < IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE; i++) {
        g_image[i] = float24_t(-888.0f);
    }
    for (int i = 0; i < FC2_WEIGHTS_W; i++) {
        g_predictions[i] = float24_t(-888.0f);
    }
    
    std::cout << "[TESTBENCH DEBUG] Initialized all arrays with recognizable values" << std::endl;

    // Measure weight loading time specifically
    auto weight_load_start = std::chrono::high_resolution_clock::now();
    
    // Load all weights
    if (!load_all_weights()) {
        std::cerr << "Failed to load weights. Please run corrected_weight_extractor_fixed.py first." << std::endl;
        return -1;
    }
    
    auto weight_load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> weight_load_time = weight_load_end - weight_load_start;

    std::cout << "\n[TESTBENCH DEBUG] ===== FINAL WEIGHT VERIFICATION BEFORE INFERENCE =====" << std::endl;
    verify_array_non_zero("FINAL conv1_weights", g_conv1_weights, CONV1_FILTERS * CONV1_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE);
    verify_array_non_zero("FINAL conv1_bias", g_conv1_bias, CONV1_FILTERS);
    verify_array_non_zero("FINAL conv2_weights", g_conv2_weights, CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE);
    verify_array_non_zero("FINAL conv2_bias", g_conv2_bias, CONV2_FILTERS);
    verify_array_non_zero("FINAL fc1_weights", g_fc1_weights, FC1_WEIGHTS_H * FC1_WEIGHTS_W);
    verify_array_non_zero("FINAL fc1_bias", g_fc1_bias, FC1_WEIGHTS_W);
    verify_array_non_zero("FINAL fc2_weights", g_fc2_weights, FC1_WEIGHTS_W * FC2_WEIGHTS_W);
    verify_array_non_zero("FINAL fc2_bias", g_fc2_bias, FC2_WEIGHTS_W);
    
    std::cout << std::endl;
    std::cout << "Available test modes:" << std::endl;
    std::cout << "1. Single test mode (test one image)" << std::endl;
    std::cout << "2. Batch test mode (test multiple images and calculate accuracy)" << std::endl;
    std::cout << std::endl;
    
    // Run batch test with 1 sample for debugging
    int num_test_samples = 5;
    // Measure batch test time
    auto batch_test_start = std::chrono::high_resolution_clock::now();
    
    run_batch_test(num_test_samples);

    auto batch_test_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> batch_test_time = batch_test_end - batch_test_start;
    
    // End timing for the entire process
    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = total_end_time - total_start_time;
    
    std::cout << std::endl;
    std::cout << "Test completed successfully!" << std::endl;
    std::cout << "Total execution time: " << total_time.count() << " ms" << std::endl;
    std::cout << "Weight loading time: " << weight_load_time.count() << " ms" << std::endl;
    std::cout << "Batch test time: " << batch_test_time.count() << " ms" << std::endl;
    std::cout << "Test completed successfully!" << std::endl;
    
    return 0;
}