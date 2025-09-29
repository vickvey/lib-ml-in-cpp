#include "csv.h"
#include <vector>
#include <iostream>
#include "../include/linear_regression.hh"
#include "../include/data_utils.hh"

void standardize(MATRIX& X) {
    size_t n_samples = X.size();
    size_t n_features = X[0].size();

    for(size_t j = 0; j < n_features; j++) {
        double mean = 0.0;
        for(size_t i = 0; i < n_samples; i++)
            mean += X[i][j];
        mean /= n_samples;

        double stddev = 0.0;
        for(size_t i = 0; i < n_samples; i++)
            stddev += (X[i][j] - mean) * (X[i][j] - mean);
        stddev = std::sqrt(stddev / n_samples);

        if(stddev < 1e-12) stddev = 1.0; // avoid division by zero

        for(size_t i = 0; i < n_samples; i++)
            X[i][j] = (X[i][j] - mean) / stddev;
    }
}

// load csv wrapper around csv.h (last column as target)
bool load_csv(const std::string& filename, MATRIX& X, VECTOR& y) {
    io::CSVReader<14> in(filename); 
    in.read_header(io::ignore_extra_column,
                   "CRIM","ZN","INDUS","CHAS","NOX","RM",
                   "AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV");

    std::string sCRIM, sZN, sINDUS, sCHAS, sNOX, sRM, sAGE, sDIS, sRAD, sTAX, sPTRATIO, sB, sLSTAT, sMEDV;

    while(in.read_row(sCRIM, sZN, sINDUS, sCHAS, sNOX, sRM, sAGE, sDIS, sRAD, sTAX, sPTRATIO, sB, sLSTAT, sMEDV)) {
        // skip row if any value is "NA"
        if (sCRIM == "NA" || sZN == "NA" || sINDUS == "NA" || sCHAS == "NA" || sNOX == "NA" ||
            sRM == "NA" || sAGE == "NA" || sDIS == "NA" || sRAD == "NA" || sTAX == "NA" ||
            sPTRATIO == "NA" || sB == "NA" || sLSTAT == "NA" || sMEDV == "NA") {
            continue;
        }

        // Making sure no datatype issues (default to double)
        double CRIM = std::stod(sCRIM), ZN = std::stod(sZN), INDUS = std::stod(sINDUS), CHAS = std::stod(sCHAS),
               NOX = std::stod(sNOX), RM = std::stod(sRM), AGE = std::stod(sAGE), DIS = std::stod(sDIS),
               RAD = std::stod(sRAD), TAX = std::stod(sTAX), PTRATIO = std::stod(sPTRATIO),
               B = std::stod(sB), LSTAT = std::stod(sLSTAT), MEDV = std::stod(sMEDV);

        VECTOR row = {CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT};
        X.push_back(row);
        y.push_back(MEDV);
    }

    return true;
}


int main() {
    MATRIX X;
    VECTOR y;

    if(!load_csv("HousingData.csv", X, y)){
        std::cerr << "Failed to load CSV!!\n";
        return 1;
    }

    std::cout << "Loaded " << X.size() << " samples with " << X[0].size() << " features\n";

    // splitting the Dataset into two parts
    TrainTestSplit split = train_test_split(X, y, 0.25, true, 42);
    standardize(split.X_train);
    standardize(split.X_test);

    // setup the model
    LinearRegression model(0.005, 10000, true);
    model.fit(split.X_train, split.y_train, 1000);
    VECTOR preds = model.predict_batch(split.X_test);

    // Diplay first 10 predictions vs actual values
    std::cout << "\nFirst 10 Predictions vs Actual:\n";
    for(size_t i = 0; i<10 && i<preds.size(); i++)
        std::cout << "Pred: " << preds[i] << " | Actual: " << split.y_test[i] << "\n";

    // some more info on stats (blah blah)
    std::cout << "\nR^2 score on test data: " << model.score(split.X_test, split.y_test) << "\n";
    std::cout << "MSE on test data: " << model.compute_loss(split.X_test, split.y_test) << "\n";

    return 0;
}
