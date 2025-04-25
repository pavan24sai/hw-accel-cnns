#include "CNNV2.h"
#include "Tensor3D.h"
#include "utils.h"
#include <chrono>

/**
 * Main function to run the optimized AlexNet CNN inference.
 */
int main(int argc, char* argv[]) {
    // Paths to the network weights and the test image.
    std::string imageFile = "../test_images/airplane.png";
    std::string weightsPath = "../weights";

    if (argc > 1) {
        imageFile = argv[1];
    }
    if (argc > 2) {
        weightsPath = argv[2];
    }

    std::cout << "Starting Optimized AlexNet CNN inference..." << std::endl;

    // Create optimized CNN model
    CNNV2 cnn;

    // Load weights
    std::cout << "Loading weights from: " << weightsPath << std::endl;
    if (!cnn.loadWeights(weightsPath)) {
        std::cerr << "Failed to load weights. Using random initialization for demonstration." << std::endl;
    }

    // Load and preprocess image
    std::cout << "Loading image: " << imageFile << std::endl;
    Tensor3D input = loadAndPreprocessImage(imageFile, 224, 224);

    // Measure time for performance comparison
    auto start = std::chrono::high_resolution_clock::now();

    // Forward pass through the network
    std::cout << "Running inference..." << std::endl;
    std::vector<float> probabilities = cnn.forward(input);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

    // Load ImageNet class labels
    std::vector<std::string> classLabels;
    std::ifstream labelFile(weightsPath + "/imagenet_classes.txt");
    std::string line;

    if (labelFile.is_open()) {
        while (std::getline(labelFile, line)) {
            classLabels.push_back(line);
        }
        labelFile.close();
    }
    else {
        std::cerr << "Warning: Could not load class labels. Using class IDs instead." << std::endl;
        for (int i = 0; i < 1000; i++) {
            classLabels.push_back("Class_" + std::to_string(i));
        }
    }

    // Get top-5 predictions
    auto top5 = cnn.getTopKPredictions(probabilities, 5);
    std::cout << "\nTop 5 predictions:" << std::endl;
    for (const auto& prediction : top5) {
        int classId = prediction.first;
        float prob = prediction.second;
        std::string className = classId < classLabels.size() ? classLabels[classId] : "Unknown";
        std::cout << className << ": " << (prob * 100.0f) << "%" << std::endl;
    }

    std::cout << "\nInference complete!" << std::endl;

    return 0;
}