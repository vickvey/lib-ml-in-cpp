#ifndef DATA_UTILS_H_
#define DATA_UTILS_H_

#include "linear_regression.hh"
#include <tuple>
#include <random>
#include <algorithm>

struct TrainTestSplit {
    MATRIX X_train;
    MATRIX X_test;
    VECTOR y_train;
    VECTOR y_test;
};

// inspired from sklearn's train_test_split
TrainTestSplit train_test_split(
    const MATRIX& X,
    const VECTOR& y,
    double test_size = 0.2,
    bool shuffle = true,
    unsigned int seed = 42
);

#endif // DATA_UTILS_H_
