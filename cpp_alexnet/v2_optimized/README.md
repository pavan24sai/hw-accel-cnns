# Version 2: Incorporating Loop Optimization Techniques

## 1. New Classes Introduced

### ConvolutionalLayerV2
This implementation of a convolutional layer incorporates loop tiling and loop reordering techniques from the research paper "[Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks](https://dl.acm.org/doi/10.1145/2684746.2689060)". The class includes:
- **Tile size parameters**: `Tm`, `Tn`, `Tr`, `Tc` for controlling parallelism and data locality.
- **TileBuffers struct**: Manages memory for local data caching.
- **Optimized forward method**: Implements tiled convolution with improved memory access patterns.
- **Support methods**: For loading/storing data and processing within optimized loop structure.

### CNNV2
The CNN implementation that incorporates the optimized convolutional layer while maintaining the original AlexNet architecture.

### Other Classes
The rest of the classes are exactly same as in the version-1 [v1_baseline](../v1_baseline/README.md).

## 2. Key Optimizations
### Loop Tiling
Divides large data structures into smaller tiles that fit into on-chip memory, improving cache locality:
```cpp
// Outer loops iterate over tiles
for (int to = 0; to < outputChannels; to += Tm) {
    for (int ti = 0; ti < inputChannels; ti += Tn) {
        for (int row = 0; row < outputHeight; row += Tr) {
            for (int col = 0; col < outputWidth; col += Tc) {
                // Process the current tile
            }
        }
    }
}
```

### Loop Reordering
Optimizes loop structure based on the paper's proposed accelerator structure (Code 3):
```cpp
// Optimized loop order: i -> j -> trr -> tcc -> too -> tii
for (int i = 0; i < kernelSize; i++) {
    for (int j = 0; j < kernelSize; j++) {
        for (int trr = 0; trr < tileRows; trr++) {
            for (int tcc = 0; tcc < tileCols; tcc++) {
                // Manual unrolling of too and tii loops
            }
        }
    }
}
```

### Manual Loop Unrolling
Simulates parallel execution without HLS pragmas:
```cpp
// Unroll too loop with step size Tm
PROCESS_TOO(0);
PROCESS_TOO(1);
PROCESS_TOO(2);
PROCESS_TOO(3);
// ...

// Unroll tii loop with step size Tn
PROCESS_TII(0);
PROCESS_TII(1);
// ... etc.
```

### Memory Access Optimization
Implements data buffering and reuse:
1. **Buffer Structure**: Dedicated buffers for input, weights, and output.
2. **Proper Accumulation**: Ensures correct result accumulation across tiles.
3. **ReLU Application**: Single application after all accumulation is complete.
This makes more sense for FPGA targeting, while running HLS.

## 3. Customization

The implementation allows customization of tile sizes to tune performance:

```cpp
// Custom tile sizes for different hardware configurations
ConvolutionalLayerV2("conv1", 3, 64, 11, 4, 2, 32, 8, 8, 8)  // Different Tm, Tn, Tr, Tc
```

This enables performance tuning based on specific hardware capabilities and memory hierarchy.