#include <vector>
#include <cstddef> 

#ifndef LINEAR_REGRESSION_API_IN_CPP_H_
#define LINEAR_REGRESSION_API_IN_CPP_H_

using MATRIX = std::vector<std::vector<double>>;
using VECTOR = std::vector<double>;
using SCALAR = double;

class LinearRegression {
public:
    explicit LinearRegression(SCALAR lr = 0.01, size_t n_iters = 1000, bool verbose = false);

    // Core API
    void fit(const MATRIX& X, const VECTOR& y, size_t log_every=100);
    SCALAR predict_one(const VECTOR& x) const;
    VECTOR predict_batch(const MATRIX& X) const;
    SCALAR score(const MATRIX& X, const VECTOR& y) const; // R^2

    // Accessors
    // VECTOR get_coef() const;      // return weights (w1, w2, ..., wn)
    // SCALAR get_intercept() const; // return bias
    // void set_params(SCALAR lr, int n_iters);
    SCALAR compute_loss(const MATRIX& X, const VECTOR& y) const; // MSE loss

    private:
    // Training helpers
    std::pair<VECTOR, SCALAR> compute_gradients(const MATRIX& X, const VECTOR& y) const;
    void update_weights(const VECTOR& grad_W, SCALAR grad_b);

    // Internal state
    VECTOR weights;   // size = n_features
    SCALAR bias;      // intercept
    SCALAR lr; // learning rate
    size_t n_iters; // total iterations
    bool verbose;
};

#endif // LINEAR_REGRESSION_API_IN_CPP_H_