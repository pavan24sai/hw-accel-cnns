#pragma once

#ifndef LAYER_H
#define LAYER_H

#include "Tensor3D.h"
#include <string>

/*
* An abstract base class defining the interface for all neural network layers. It enforces a common protocol with the forward() method, 
* enabling data to flow through the network in a consistent manner. Each derived layer implements its specific forward pass computation 
* while maintaining a unified interface for the network.
*/

class Layer {
protected:
    std::string name;

public:
    Layer(const std::string& name);
    virtual ~Layer();

    virtual Tensor3D forward(const Tensor3D& input) = 0;
    virtual bool loadWeights(const std::string& filename);

    std::string getName() const;
};

#endif // LAYER_H
