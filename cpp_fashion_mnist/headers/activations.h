/******************************************************************************
 * Fashion MNIST CNN Activation Functions - Minimalist Version for Zybo Z7-20
 * Only essential functions with minimal resource usage
 ******************************************************************************/

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "defines.h"

// Add inline keyword to prevent multiple definition errors
inline float24_t relu(float24_t x) {
    return (x > float24_t(0)) ? x : float24_t(0);
}

// Add inline keyword to prevent multiple definition errors
inline float24_t max_pool(float24_t a, float24_t b) {
    return (a > b) ? a : b;
}

#endif // ACTIVATIONS_H