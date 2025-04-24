#include "Layer.h"

Layer::Layer(const std::string& name) : name(name) {}
Layer::~Layer() {}

bool Layer::loadWeights(const std::string& filename) {
    return true;
}

std::string Layer::getName() const {
    return name;
}
