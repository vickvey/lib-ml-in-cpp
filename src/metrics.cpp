#include <stdexcept> 
#include "../include/metrics.hh"
#include <cmath>

SCALAR mean_squared_error(const VECTOR& y_true, const VECTOR& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same length");
    }

    SCALAR mse = 0.0;
    for (size_t i = 0; i < y_true.size(); i++) {
        SCALAR diff = y_true[i] - y_pred[i];
        mse += diff * diff;
    }
    return mse / y_true.size();
}
