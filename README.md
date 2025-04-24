# hw-accel-cnns
Hardware Accelerator Design for CNNs on FPGAs.
Target FPGA Board: AMD Kria SOM KV260 Vision AI Starter Kit

## Pre-Trained AlexNet Model Implementation in C++
[cpp_alexnet](./cpp_alexnet) contains the class-based C++ implementation of the AlexNet model.
- The [alexnet_model_weights_extract](./alexnet_model_weights_extract.py) script is used to download the model weights from the **torch** library.
- The baseline model is implemented in C++, to understand the end-to-end inference pipeline for AlexNet architecture.
- Other dependencies:
  - The **stb_image.h** header file is downloaded from https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
  - The ImageNet class labels are downloaded from https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
- The idea is to optimize this implementation for FPGA targeting, by leveraging High-Level Synthesis (HLS) workflows.