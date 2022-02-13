import sys
import progressbar
import numpy as np

sys.path.insert(1, '../Utils/')

from regressorsUtils import *
from dataProcessor import normalize
from dataProcessingUtils import split_data
from comparisonUtils import find_max_avg

# Train each algorithm n times and get avg mse
def compare_algorinthms_mse(iterations=100):
    data = normalize('../Datasets/house_data_2022-01-26.csv')[0]
    X_train, X_test, y_train, y_test = split_data(data, 0.2)

    bayesian_mse, svr_mse, rf_mse, knn_mse = [], [], [], []

    print('Starting model comparison')
    bar = progressbar.ProgressBar(maxval=iterations).start()

    for i in range(iterations):
        b_mse = runBayesianRidge(X_train, X_test, y_train, y_test)[1]
        s_mse = runSupportVector(X_train, X_test, y_train, y_test)[1]
        r_mse = runRandomForest(X_train, X_test, y_train, y_test)[1]
        k_mse = runKNeighbors(X_train, X_test, y_train, y_test)[1]

        bayesian_mse.append(b_mse)
        svr_mse.append(s_mse)
        rf_mse.append(r_mse)
        knn_mse.append(k_mse)
        bar.update(i)

    print('\nModel comparison successfully terminated')
    best_list, best_avg = find_max_avg([bayesian_mse, svr_mse, rf_mse, knn_mse])
    if best_list == bayesian_mse:
        return 'Bayesian', best_avg
    elif best_list == svr_mse:
        return 'SVR', best_avg
    elif best_list == rf_mse:
        return 'Random Forest', best_avg
    elif best_list == knn_mse:
        return 'KNN', best_avg

# Train each algorithm n times and get avg mse
def compare_algorinthms_mse(iterations=10):
    data = normalize('../Datasets/house_data_2022-01-26.csv')[0]
    X_train, X_test, y_train, y_test = split_data(data, 0.2)

    bayesian_mse, svr_mse, rf_r, knn_mse = [], [], [], []

    print('Starting model comparison')
    bar = progressbar.ProgressBar(maxval=iterations).start()

    for i in range(iterations):
        b_mse = runBayesianRidge(X_train, X_test, y_train, y_test)[1]
        s_mse = runSupportVector(X_train, X_test, y_train, y_test)[1]
        r_mse = runRandomForest(X_train, X_test, y_train, y_test)[1]
        k_mse = runKNeighbors(X_train, X_test, y_train, y_test)[1]

        bayesian_mse.append(b_mse)
        svr_mse.append(s_mse)
        rf_mse.append(r_mse)
        knn_mse.append(k_mse)
        bar.update(i)

    print('\nModel comparison successfully terminated')
    best_list, best_avg = find_max_avg([bayesian_mse, svr_mse, rf_mse, knn_mse])
    if best_list == bayesian_mse:
        return 'Bayesian', best_avg
    elif best_list == svr_mse:
        return 'SVR', best_avg
    elif best_list == rf_mse:
        return 'Random Forest', best_avg
    elif best_list == knn_mse:
        return 'KNN', best_avg
    
def main(type_of_comp, itrs):
    if type_of_comp == 'mse':
        best_algorithm, best_avg = compare_algorinthms_mse(iterations=itrs)
        print('{} is the best algorithms after {} iterations with MSE of {}'.format(best_algorithm, str(itrs), str(round(best_avg, 3))))


if __name__ == "__main__":
    main('mse', 100)