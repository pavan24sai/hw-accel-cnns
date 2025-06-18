# hw-accel-cnns
Hardware Accelerator Design for CNNs on FPGAs. **This is an AI-assisted project.**

## Project Overview
The end-to-end project is explained here: [CNN_Accelerator_Design_Document](./CNN_Accelerator_Design_Document.pdf).
For a quick overview, take a look at the [CNN_Accelerator_PPT](./CNN_Accelerator_PPT.pdf) document.
Target FPGA Board: AMD Kria SOM KV260 Vision AI Starter Kit.

## Codebase Overview
1. [cpp_alexnet](./cpp_alexnet) - C++ implementation of AlexNet CNN for object detection.
2. [hw-accel-cnns](./hw-accel-cnns) - C++ HLS implementation of a custom CNN designed for MNIST fashion dataset classification.

## Results
1. The implementation achieves 9.32ms inference time while consuming only 3.515W in programmable logic, providing 1.4× better energy efficiency compared to CPU execution. 
2. The 16-bit fixed-point implementation maintains 91% classification accuracy, representing only 2.7% degradation from floating-point reference.
3. The HLS IP and the hardware bitstream files can be found in the [results](./results) folder.