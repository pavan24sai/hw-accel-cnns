#include "ConvolutionalLayerV2.h"

ConvolutionalLayerV2::TileBuffers::TileBuffers(int tm, int tn, int tr, int tc, int k, int s)
    : tileRows(tr), tileCols(tc), tiSize(tn), toSize(tm) {
    tileInputHeight = tr * s + k - s;
    tileInputWidth = tc * s + k - s;

    inputBuffer.resize(tn * tileInputHeight * tileInputWidth, 0.0f);
    weightBuffer.resize(tm * tn * k * k, 0.0f);
    outputBuffer.resize(tm * tr * tc, 0.0f);
}

ConvolutionalLayerV2::ConvolutionalLayerV2(const std::string& name,
    int inputChannels,
    int outputChannels,
    int kernelSize,
    int stride,
    int padding,
    int tileSizeM,
    int tileSizeN,
    int tileSizeR,
    int tileSizeC)
    : Layer(name),
    inputChannels(inputChannels),
    outputChannels(outputChannels),
    kernelSize(kernelSize),
    stride(stride),
    padding(padding),
    weights(outputChannels, inputChannels, kernelSize* kernelSize),
    bias(outputChannels, 0.0f),
    Tm(tileSizeM),
    Tn(tileSizeN),
    Tr(tileSizeR),
    Tc(tileSizeC) {
}

Tensor3D ConvolutionalLayerV2::forward(const Tensor3D& input) {
    int inputHeight = input.getHeight();
    int inputWidth = input.getWidth();

    // Calculate output dimensions based on input, kernel size, stride and padding
    int outputHeight = ((inputHeight + 2 * padding - kernelSize) / stride) + 1;
    int outputWidth = ((inputWidth + 2 * padding - kernelSize) / stride) + 1;

    // Create output tensor initialized with bias
    Tensor3D output(outputChannels, outputHeight, outputWidth);

    // Initialize with bias ONCE (not in each tile)
    for (int to = 0; to < outputChannels; to++) {
        for (int row = 0; row < outputHeight; row++) {
            for (int col = 0; col < outputWidth; col++) {
                output.at(to, row, col) = bias[to];
            }
        }
    }

    // Perform tiled convolution operation
    for (int to = 0; to < outputChannels; to += Tm) {
        int toLimit = std::min(to + Tm, outputChannels);

        for (int ti = 0; ti < inputChannels; ti += Tn) {
            int tiLimit = std::min(ti + Tn, inputChannels);

            for (int row = 0; row < outputHeight; row += Tr) {
                int rowLimit = std::min(row + Tr, outputHeight);

                for (int col = 0; col < outputWidth; col += Tc) {
                    int colLimit = std::min(col + Tc, outputWidth);

                    // Allocate buffers for the current tile
                    TileBuffers buffers(
                        toLimit - to,
                        tiLimit - ti,
                        rowLimit - row,
                        colLimit - col,
                        kernelSize,
                        stride
                    );

                    // Load input tile data with padding handling
                    loadInputTile(input, buffers, ti, tiLimit, row, rowLimit, col, colLimit);

                    // Load weight tile data
                    loadWeightTile(buffers, to, toLimit, ti, tiLimit);

                    // Initialize output buffer with ZEROS (not bias)
                    initOutputTileZero(buffers);

                    // Process tile using optimized loop ordering
                    processTile(buffers, ti, tiLimit, to, toLimit);

                    // ACCUMULATE to output tensor (not overwrite)
                    accumulateOutputTile(output, buffers, to, toLimit, row, rowLimit, col, colLimit);
                }
            }
        }
    }

    // Apply ReLU activation ONCE at the end (not per tile)
    for (int to = 0; to < outputChannels; to++) {
        for (int row = 0; row < outputHeight; row++) {
            for (int col = 0; col < outputWidth; col++) {
                output.at(to, row, col) = std::max(0.0f, output.at(to, row, col));
            }
        }
    }

    return output;
}

void ConvolutionalLayerV2::initializeWeights(float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, stddev);

    // Initialize weights with small random values
    for (int to = 0; to < outputChannels; to++) {
        for (int ti = 0; ti < inputChannels; ti++) {
            for (int k = 0; k < kernelSize * kernelSize; k++) {
                weights.at(to, ti, k) = dist(gen);
            }
        }
        // Initialize bias
        bias[to] = dist(gen);
    }
}

bool ConvolutionalLayerV2::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open weight file: " << filename << std::endl;
        return false;
    }

    // Get file size
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate expected size for weights and bias
    size_t weightsSize = outputChannels * inputChannels * kernelSize * kernelSize * sizeof(float);
    size_t biasSize = outputChannels * sizeof(float);

    if (fileSize != weightsSize + biasSize) {
        std::cerr << "Error: File size mismatch for " << filename << std::endl;
        std::cerr << "Expected: " << weightsSize + biasSize << " bytes, Got: " << fileSize << " bytes" << std::endl;
        return false;
    }

    // Read weights
    std::vector<float> weightsData(outputChannels * inputChannels * kernelSize * kernelSize);
    file.read(reinterpret_cast<char*>(weightsData.data()), weightsSize);

    // Reshape and copy weights
    for (int to = 0; to < outputChannels; to++) {
        for (int ti = 0; ti < inputChannels; ti++) {
            for (int k = 0; k < kernelSize * kernelSize; k++) {
                int idx = (to * inputChannels * kernelSize * kernelSize) +
                    (ti * kernelSize * kernelSize) + k;
                weights.at(to, ti, k) = weightsData[idx];
            }
        }
    }

    // Read bias
    file.read(reinterpret_cast<char*>(bias.data()), biasSize);

    std::cout << "Successfully loaded weights and bias for layer with shapes: ["
        << outputChannels << ", " << inputChannels << ", " << kernelSize << "x" << kernelSize << "]" << std::endl;

    return true;
}

void ConvolutionalLayerV2::loadInputTile(const Tensor3D& input, TileBuffers& buffers,
    int tiStart, int tiEnd, int rowStart, int rowEnd, int colStart, int colEnd) {
    int inputHeight = input.getHeight();
    int inputWidth = input.getWidth();

    // For each input channel in the tile
    for (int tii = tiStart; tii < tiEnd; tii++) {
        int tiiOffset = (tii - tiStart) * buffers.tileInputHeight * buffers.tileInputWidth;

        // For each position in the input tile
        for (int h = 0; h < buffers.tileInputHeight; h++) {
            for (int w = 0; w < buffers.tileInputWidth; w++) {
                // Calculate position in the original input
                int inputRow = rowStart * stride + h - padding;
                int inputCol = colStart * stride + w - padding;

                // Handle padding
                if (inputRow >= 0 && inputRow < inputHeight &&
                    inputCol >= 0 && inputCol < inputWidth) {
                    buffers.inputBuffer[tiiOffset + h * buffers.tileInputWidth + w] = input.at(tii, inputRow, inputCol);
                }
                else {
                    buffers.inputBuffer[tiiOffset + h * buffers.tileInputWidth + w] = 0.0f; // Zero padding
                }
            }
        }
    }
}

void ConvolutionalLayerV2::loadWeightTile(TileBuffers& buffers, int toStart, int toEnd, int tiStart, int tiEnd) {
    for (int too = toStart; too < toEnd; too++) {
        int tooOffset = (too - toStart) * buffers.tiSize * kernelSize * kernelSize;

        for (int tii = tiStart; tii < tiEnd; tii++) {
            int tiiOffset = tooOffset + (tii - tiStart) * kernelSize * kernelSize;

            for (int i = 0; i < kernelSize; i++) {
                for (int j = 0; j < kernelSize; j++) {
                    int weightIdx = i * kernelSize + j;
                    buffers.weightBuffer[tiiOffset + weightIdx] = weights.at(too, tii, weightIdx);
                }
            }
        }
    }
}

void ConvolutionalLayerV2::initOutputTile(TileBuffers& buffers, int toStart, int toEnd) {
    for (int too = toStart; too < toEnd; too++) {
        int tooOffset = (too - toStart) * buffers.tileRows * buffers.tileCols;
        float biasVal = bias[too];

        for (int trr = 0; trr < buffers.tileRows; trr++) {
            for (int tcc = 0; tcc < buffers.tileCols; tcc++) {
                buffers.outputBuffer[tooOffset + trr * buffers.tileCols + tcc] = biasVal;
            }
        }
    }
}

void ConvolutionalLayerV2::processTile(TileBuffers& buffers, int tiStart, int tiEnd, int toStart, int toEnd) {
    // Following the paper's proposed accelerator structure (Code 3):
        // Loop order: i -> j -> trr -> tcc -> too -> tii
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            for (int trr = 0; trr < buffers.tileRows; trr++) {
                for (int tcc = 0; tcc < buffers.tileCols; tcc++) {
                    // Manual unrolling of too loop (simulating parallelism)
                    computeTileUnitToo(buffers, i, j, trr, tcc, tiStart, tiEnd, toStart, toEnd);
                }
            }
        }
    }
}

void ConvolutionalLayerV2::computeTileUnitToo(TileBuffers& buffers, int i, int j, int trr, int tcc,
    int tiStart, int tiEnd, int toStart, int toEnd) {
    // Manual unrolling with step size Tm
        // This simulates parallel processing of Tm output channels
    int tooLimit = std::min(Tm, toEnd - toStart);

#define PROCESS_TOO(index) \
        if ((toStart + index) < toEnd) { \
            computeTileUnitTii(buffers, i, j, trr, tcc, tiStart, tiEnd, toStart + index, toStart); \
        }

    // Unroll based on typical Tm values (from the paper, Tm=64 worked well)
    PROCESS_TOO(0);
    PROCESS_TOO(1);
    PROCESS_TOO(2);
    PROCESS_TOO(3);
    // For a full implementation, generate all iterations or use dynamic approach

    // For remaining iterations that don't fit the unroll factor
    for (int tooIdx = 4; tooIdx < tooLimit; tooIdx++) {
        PROCESS_TOO(tooIdx);
    }

#undef PROCESS_TOO
}

void ConvolutionalLayerV2::computeTileUnitTii(TileBuffers& buffers, int i, int j, int trr, int tcc,
    int tiStart, int tiEnd, int too, int toStartOffset) {
    int tooOffset = (too - toStartOffset) * buffers.tileRows * buffers.tileCols;
    int weightIdxBase = (too - toStartOffset) * buffers.tiSize * kernelSize * kernelSize;
    int outputIdx = tooOffset + trr * buffers.tileCols + tcc;

    // Manual unrolling with step size Tn
    // This simulates parallel processing of Tn input channels
    int tiiLimit = std::min(Tn, tiEnd - tiStart);

#define PROCESS_TII(index) \
        if ((tiStart + index) < tiEnd) { \
            int tii = tiStart + index; \
            /* Correctly calculate weight index */ \
            float weight = buffers.weightBuffer[weightIdxBase + (tii - tiStart) * kernelSize * kernelSize + i * kernelSize + j]; \
            /* Correctly calculate input index */ \
            int h = trr * stride + i; \
            int w = tcc * stride + j; \
            int inputIdx = (tii - tiStart) * buffers.tileInputHeight * buffers.tileInputWidth + \
                          h * buffers.tileInputWidth + w; \
            /* Accumulate to output */ \
            buffers.outputBuffer[outputIdx] += weight * buffers.inputBuffer[inputIdx]; \
        }

    // Unroll based on typical Tn values (from the paper, Tn=7 worked well)
    PROCESS_TII(0);
    PROCESS_TII(1);
    PROCESS_TII(2);
    PROCESS_TII(3);
    PROCESS_TII(4);
    PROCESS_TII(5);
    PROCESS_TII(6);

    // For remaining iterations that don't fit the unroll factor
    for (int tiiIdx = 7; tiiIdx < tiiLimit; tiiIdx++) {
        PROCESS_TII(tiiIdx);
    }

#undef PROCESS_TII
}

void ConvolutionalLayerV2::initOutputTileZero(TileBuffers& buffers) {
    std::fill(buffers.outputBuffer.begin(), buffers.outputBuffer.end(), 0.0f);
}

void ConvolutionalLayerV2::accumulateOutputTile(Tensor3D& output, TileBuffers& buffers,
    int toStart, int toEnd, int rowStart, int rowEnd, int colStart, int colEnd) {
    for (int too = toStart; too < toEnd; too++) {
        int tooOffset = (too - toStart) * buffers.tileRows * buffers.tileCols;

        for (int trr = 0; trr < buffers.tileRows; trr++) {
            for (int tcc = 0; tcc < buffers.tileCols; tcc++) {
                float val = buffers.outputBuffer[tooOffset + trr * buffers.tileCols + tcc];
                // ACCUMULATE instead of overwrite
                output.at(too, rowStart + trr, colStart + tcc) += val;
            }
        }
    }
}