#include "../include/data_utils.hh"

TrainTestSplit train_test_split(
    const MATRIX& X,
    const VECTOR& y,
    double test_size,
    bool shuffle,
    unsigned int seed
) {
    size_t n_samples = X.size();
    size_t n_test = static_cast<size_t>(n_samples * test_size);
    size_t n_train = n_samples - n_test;

    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    TrainTestSplit split;

    for (size_t i = 0; i < n_train; i++) {
        split.X_train.push_back(X[indices[i]]);
        split.y_train.push_back(y[indices[i]]);
    }
    for (size_t i = n_train; i < n_samples; i++) {
        split.X_test.push_back(X[indices[i]]);
        split.y_test.push_back(y[indices[i]]);
    }

    return split;
}
