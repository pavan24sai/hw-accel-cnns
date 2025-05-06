## Pre-Trained AlexNet Model Implementation in C++
This repo contains the class-based C++ implementation of the AlexNet model.
- The [alexnet_model_weights_extract](./alexnet_model_weights_extract.py) script is used to download the model weights from the **torch** library.
- Other dependencies:
  - The **stb_image.h** header file is downloaded from https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
  - The ImageNet class labels are downloaded from https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
- The idea is to optimize this implementation for FPGA targeting, by leveraging High-Level Synthesis (HLS) workflows.

## Version History
### 3. v3_hls_compatible
[v3_hls_compatible](./v3_hls_compatible) implements the HLS compatible C++ code for generating the RTL IP for the accelerator. It contains all the necessary HLS_PRAGMAS to generate the RTL IP as required. Vitis HLS is the tool used for running the HLS.

### 2. v2_optimized
[v2_optimized](./v2_optimized) implements an optimized version by incorporating the techniques illustrated in "[Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks](https://dl.acm.org/doi/10.1145/2684746.2689060)".

### 1. v1_baseline
[v1_baseline](./v1_baseline) implements an initial version of the AlexNet in C++. This is a class-based implementation.