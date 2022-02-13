import sys

sys.path.insert(1, 'Utils/')
sys.path.insert(1, 'Processes/')

from dataProcessor import normalize
from regressorsUtils import *
from predictionUtils import *
from dataProcessingUtils import split_data

def etestRFR(itterations=100):
    data, scalers = normalize('Datasets/house_data_2022-01-26.csv', scaled_list = ['price','year','loc', 'type'],encoding_type='label')
    X_train, X_test, y_train, y_test = split_data(data, 0.2)
    testRFR(X_train, X_test, y_train, y_test, its=itterations)

def ctestSVR(itterations=100):
    data, scalers = normalize('Datasets/house_data_2022-01-26.csv', scaled_list = ['price','year','loc', 'type'],encoding_type='label')
    X_train, X_test, y_train, y_test = split_data(data, 0.2)
    testSVR(X_train, X_test, y_train, y_test, its=itterations)

def ntestBRR(itterations=100):
    data, scalers = normalize('Datasets/house_data_2022-01-26.csv', scaled_list = ['price','year','loc', 'type'],encoding_type='label')
    X_train, X_test, y_train, y_test = split_data(data, 0.2)
    testBRR(X_train, X_test, y_train, y_test, its=itterations)

def ktestKNN(itterations=20):
    data, scalers = normalize('Datasets/house_data_2022-01-26.csv', scaled_list = ['price','year','loc', 'type'],encoding_type='label')
    X_train, X_test, y_train, y_test = split_data(data, 0.2)
    testKNN(X_train, X_test, y_train, y_test, its=itterations)

def main(encoding_type='one_hot', show_plots=False, scaling='min_max'):
    if encoding_type == 'label':
        print('Encoding Type is set to Label')
        data, scalers = normalize('Datasets/house_data_2022-01-26.csv', scaled_list = ['price','year','loc', 'type'],encoding_type='label')
    elif scaling == 'standard':
        print('Results with Standard Scaling')
        data, scalers = normalize('Datasets/house_data_2022-01-26.csv', scaling_method = 'standard')
    elif scaling == 'log':
        print('Results with Log Transformation')
        data, scalers = normalize('Datasets/house_data_2022-01-26.csv', scaling_method = 'log')
    else:
        print('Encoding Type is set to one_hot and Scaling to min_max')
        data, scalers = normalize('Datasets/house_data_2022-01-26.csv')



    X_train, X_test, y_train, y_test = split_data(data, 0.2)

    bayesian_clf, bayesian_mse, bayesian_rs, bayesian_rmse, bayesian_plt = runBayesianRidge(X_train, X_test, y_train, y_test)
    print_model_results('Bayesian Ridge Regressor', bayesian_mse, bayesian_rs, bayesian_rmse)


 
    svm_rgs, svm_mse, svm_rs,svm_rmse, svm_plt = runSupportVector(X_train, X_test, y_train, y_test)
    print_model_results('Support Vector Regressor', svm_mse, svm_rs, svm_rmse)
    
    rf_rgs, rf_mse, rf_rs, rf_rmse, rf_plt = runRandomForest(X_train, X_test, y_train, y_test)
    print_model_results('Random Forest Regressor', rf_mse, rf_rs, rf_rmse)

    kn_rgs, kn_mse, kn_rs, kn_rmse, kn_plt = runKNeighbors(X_train, X_test, y_train, y_test)
    print_model_results('K-Neighbors Regressor', kn_mse, kn_rs, kn_rmse)


    lr_rgs, lr_mse, lr_rs, lr_rmse, lr_plt = runLinearRegression(X_train, X_test, y_train, y_test)


    print_model_results('Linear Regressor', lr_mse, lr_rs, lr_rmse)
    if show_plots:  
        bayesian_plt.show()
        svm_plt.show()
        rf_plt.show()
        kn_plt.show()
        lr_plt.show()

if __name__ == "__main__":
    if sys.argv[1] == 'default':
        main()
    elif sys.argv[1] == 'scaling' and sys.argv[2] == 'log':
        main(scaling='log')
    elif sys.argv[1] == 'encoding' and sys.argv[2] == 'lablel':
        main(encoding_type='label')
    elif sys.argv[1] == 'scaling' and sys.argv[2] == 'standard':
        main(scaling='standard')
    elif sys.argv[1] == 'test' and sys.argv[2] == 'knn':
        ktestKNN()
    elif sys.argv[1] == 'test' and sys.argv[2] == 'brr':
        ntestBRR()
    elif sys.argv[1] == 'test' and sys.argv[2] == 'svr':
        ctestSVR()
    elif sys.argv[1] == 'test' and sys.argv[2] == 'rfr':
        etestRFR()
    else:
        main()
