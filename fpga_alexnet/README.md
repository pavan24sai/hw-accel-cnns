# FPGA Files

This repository contains the file for targeting the MNIST CNN accelerator onto the Kria KV260 board.

1. [fashion_mnist_cnn_accelerator](./fashion_mnist_cnn_accelerator.zip) - Zip file containing the HLS IP for CNN acceleration. Generated using Vitis HLS. The HLS compatible C++ code can be found in [v3_hls_compatible](../cpp_alexnet/v3_hls_compatible) folder.
2. [mnist_sd_card_files](./mnist_sd_card_files.zip) - Zip file containing the files to be loaded onto the SD card for running the baremetal host application (.elf) for MNIST fashion dataset classification on the PS. The application leverages the PL bitstream for CNN compute acceleration.