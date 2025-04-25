#pragma once

#include "Layer.h"
#include "Tensor3D.h"
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <iostream>
#include <algorithm>

/**
 * Optimized ConvolutionalLayer implementation following the loop optimization
 * strategies from "Optimizing FPGA-based Accelerator Design for Deep
 * Convolutional Neural Networks" by Chen Zhang et al.
 */
class ConvolutionalLayerV2 : public Layer {
private:
    int inputChannels;     // N in the paper
    int outputChannels;    // M in the paper
    int kernelSize;        // K in the paper
    int stride;            // S in the paper
    int padding;           // P (not in the original paper code)
    Tensor3D weights;      // Weights[M][N][K][K]
    std::vector<float> bias; // Bias terms

    // Tile size parameters
    int Tm; // Tile size for output feature maps
    int Tn; // Tile size for input feature maps
    int Tr; // Tile size for output rows
    int Tc; // Tile size for output columns

    // Helper class for input and weight buffers to simulate optimized memory access
    struct TileBuffers {
        std::vector<float> inputBuffer;  // [Tn][TrxS+K-S][TcxS+K-S]
        std::vector<float> weightBuffer; // [Tm][Tn][K][K]
        std::vector<float> outputBuffer; // [Tm][Tr][Tc]
        int tileRows;                   // Number of rows in this tile
        int tileCols;                   // Number of columns in this tile
        int tileInputHeight;            // Height of input tile including kernel overlap
        int tileInputWidth;             // Width of input tile including kernel overlap
        int tiSize;                     // Number of input channels in this tile
        int toSize;                     // Number of output channels in this tile

        TileBuffers(int tm, int tn, int tr, int tc, int k, int s);
    };

public:
    ConvolutionalLayerV2(const std::string& name,
        int inputChannels,
        int outputChannels,
        int kernelSize,
        int stride,
        int padding = 0,
        int tileSizeM = 64,
        int tileSizeN = 7,
        int tileSizeR = 16,
        int tileSizeC = 16);

    virtual Tensor3D forward(const Tensor3D& input) override;
    void initializeWeights(float stddev = 0.01f);
    virtual bool loadWeights(const std::string& filename) override;

private:
    void loadInputTile(const Tensor3D& input, TileBuffers& buffers,
        int tiStart, int tiEnd, int rowStart, int rowEnd, int colStart, int colEnd);

    void loadWeightTile(TileBuffers& buffers, int toStart, int toEnd, int tiStart, int tiEnd);

    void initOutputTile(TileBuffers& buffers, int toStart, int toEnd);

    void processTile(TileBuffers& buffers, int tiStart, int tiEnd, int toStart, int toEnd);

    void computeTileUnitToo(TileBuffers& buffers, int i, int j, int trr, int tcc,
        int tiStart, int tiEnd, int toStart, int toEnd);

    void computeTileUnitTii(TileBuffers& buffers, int i, int j, int trr, int tcc,
        int tiStart, int tiEnd, int too, int toStartOffset);

    void initOutputTileZero(TileBuffers& buffers);

    void accumulateOutputTile(Tensor3D& output, TileBuffers& buffers,
        int toStart, int toEnd, int rowStart, int rowEnd, int colStart, int colEnd);
};
