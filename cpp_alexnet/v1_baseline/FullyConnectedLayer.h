#pragma once

#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include "Layer.h"
#include <vector>
#include <random>
#include <fstream>
#include <iostream>

/*
* Implements a traditional neural network layer where each neuron connects to all neurons in the previous layer. 
* These layers appear at the end of the network and transform the spatially organized features into class probabilities. 
* They contain the majority of the model's parameters and perform matrix multiplication between inputs and weights.
*/

class FullyConnectedLayer : public Layer {
private:
    int inputSize;
    int outputSize;
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;

public:
    FullyConnectedLayer(const std::string& name, int inputSize, int outputSize);

    virtual Tensor3D forward(const Tensor3D& input) override;
    void initializeWeights(float stddev = 0.01f);
    virtual bool loadWeights(const std::string& filename) override;
};

#endif // FULLYCONNECTEDLAYER_H