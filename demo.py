import sys

sys.path.insert(1, 'Utils/')
sys.path.insert(1, 'Processes/')

from dataProcessor import normalize
from regressorsUtils import *
from dataProcessingUtils import split_data

def main():
    
    data, scalers = normalize('Datasets/house_data_2022-01-26.csv', outlier_lower = .02, outlier_upper= .98)
    X_train, X_test, y_train, y_test = split_data(data, 0.2)

    bayesian_clf, bayesian_mse, bayesian_rs, bayesian_plt = runBayesianRidge(X_train, X_test, y_train, y_test)
    print_model_results('Bayesian Ridge Regressor', bayesian_mse, bayesian_rs)
    bayesian_plt.show()

    svm_rgs, svm_mse, svm_rs, svm_plt = runSupportVector(X_train, X_test, y_train, y_test)
    print_model_results('Support Vector Regressor', svm_mse, svm_rs)
    svm_plt.show()

    rf_rgs, rf_mse, rf_rs, rf_plt = runRandomForest(X_train, X_test, y_train, y_test)
    print_model_results('Random Forest Regressor', rf_mse, rf_rs)
    rf_plt.show()

    kn_rgs, kn_mse, kn_rs, kn_plt = runKNeighbors(X_train, X_test, y_train, y_test)
    print_model_results('K-Neighbors Regressor', kn_mse, kn_rs)
    kn_plt.show()

    lr_rgs, lr_mse, lr_rs, lr_plt = runLinearRegression(X_train, X_test, y_train, y_test)
    print_model_results('Linear Regressor', lr_mse, lr_rs)
    lr_plt.show()

if __name__ == "__main__":
   main()