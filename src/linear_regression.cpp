#include "../include/linear_regression.hh"
#include <cmath>
#include <numeric>
#include <iostream>

LinearRegression::LinearRegression(SCALAR lr, size_t n_iters, bool verbose): lr(lr), n_iters(n_iters), bias(0.0), verbose(verbose) {}

// simple dot product
inline SCALAR dot(const VECTOR& a, const VECTOR& b) {
    SCALAR result = 0.0;
    for (size_t i = 0; i < a.size(); i++) result += a[i] * b[i];
    return result;
}

void LinearRegression::fit(const MATRIX& X, const VECTOR& y, size_t log_every) {
    size_t n_samples = X.size();
    size_t n_features = X[0].size();
    weights.assign(n_features, 0.0);
    bias = 0.0;

    for (size_t iter = 0; iter < n_iters; iter++) {
        auto [grad_W, grad_B] = compute_gradients(X, y);
        update_weights(grad_W, grad_B);

        if (verbose && iter % log_every == 0) {
            SCALAR mse = compute_loss(X, y);
            std::cout << "[Iter " << iter << "] "
                      << "MSE: " << mse << " | Bias: " << bias;
            // std::cout <<  " | Weights: ";
            // for (auto w : weights) std::cout << w << " ";
            std::cout << std::endl;
        }
    }
}

// Predict one sample
SCALAR LinearRegression::predict_one(const VECTOR& x) const {
    SCALAR y_hat = bias;
    for (size_t j = 0; j < x.size(); j++) {
        y_hat += weights[j] * x[j];
    }
    return y_hat;
}

// Predict batch (batch size we can decide)
VECTOR LinearRegression::predict_batch(const MATRIX& X) const {
    VECTOR preds(X.size());
    for (size_t i = 0; i < X.size(); i++) {
        SCALAR y_hat = bias;
        for (size_t j = 0; j < X[i].size(); j++)
            y_hat += weights[j] * X[i][j];
        preds[i] = y_hat;
    }
    return preds;
}

// Computing gradients usng standad gradient descent
std::pair<VECTOR, SCALAR> LinearRegression::compute_gradients(const MATRIX& X, const VECTOR& y) const {
    size_t n_samples = X.size();
    size_t n_features = X[0].size();

    VECTOR gradW(n_features, 0.0);
    SCALAR gradB = 0.0;

    for (size_t i = 0; i < n_samples; i++) {
        SCALAR y_hat = bias;
        for (size_t j = 0; j < n_features; j++)
            y_hat += weights[j] * X[i][j];

        SCALAR error = y_hat - y[i];

        for (size_t j = 0; j < n_features; j++)
            gradW[j] += error * X[i][j];

        gradB += error;
    }

    for (size_t j = 0; j < n_features; j++) gradW[j] /= n_samples;
    gradB /= n_samples;

    return {gradW, gradB};
}

void LinearRegression::update_weights(const VECTOR& gradW, SCALAR gradB) {
    for (size_t j = 0; j < weights.size(); j++) {
        weights[j] -= lr * gradW[j];
    }
    bias -= lr * gradB;
}

// R^2 Score
SCALAR LinearRegression::score(const MATRIX& X, const VECTOR& y) const {
    VECTOR preds = predict_batch(X);

    // Thanks to online article on investopedia
    SCALAR ss_res = 0.0; // residual sum of squares
    SCALAR ss_tot = 0.0; // total sum of squares

    // Compute mean of y
    SCALAR y_mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();

    for (size_t i = 0; i < y.size(); i++) {
        ss_res += (y[i] - preds[i]) * (y[i] - preds[i]);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }

    return 1 - (ss_res / ss_tot);
}

// mse here
SCALAR LinearRegression::compute_loss(const MATRIX& X, const VECTOR& y) const {
    VECTOR preds = predict_batch(X);
    SCALAR loss = 0.0;
    for (size_t i = 0; i < y.size(); i++) {
        SCALAR diff = preds[i] - y[i];
        loss += diff * diff;
    }
    return loss / y.size();
}