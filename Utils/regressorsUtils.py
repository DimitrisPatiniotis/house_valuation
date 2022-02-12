# Graphs
import matplotlib.pyplot as plt

# Models
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Metrics
from sklearn.metrics import mean_squared_error, r2_score

def print_model_results(model_name, mse, rs, rmse):
    print('{} has an MSE of {},a R-squared of {} and a Root MSE of {}'.format(model_name, round(mse, 3), round(rs,3), round(rmse,3)))

def test_vs_pred_plot(y_test,y_pred, title_text=''):
    x_ax = range(len(y_test))
    plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
    plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.title(title_text)
    return plt

def runBayesianRidge(X_train, X_test, y_train, y_test):
    clf = BayesianRidge(n_iter=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mse, rs, rmse = mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred, squared=False)
    plt = test_vs_pred_plot(y_test,y_pred, 'Bayesian Ridge Regressor Accuracy')
    return clf, mse, rs, rmse, plt

def runSupportVector(X_train, X_test, y_train, y_test):
    svr = SVR()
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    mse, rs, rmse = mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred, squared=False)
    plt = test_vs_pred_plot(y_test,y_pred, 'Support Vector Regressor Accuracy')
    return svr, mse, rs, rmse, plt

def runRandomForest(X_train, X_test, y_train, y_test):
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse, rs, rmse = mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred, squared=False)
    plt = test_vs_pred_plot(y_test,y_pred, 'Random Forest Regressor Accuracy')

    return rf, mse, rs, rmse, plt

def runKNeighbors(X_train, X_test, y_train, y_test):
    kn = KNeighborsRegressor()
    kn.fit(X_train, y_train)
    y_pred = kn.predict(X_test)
    mse, rs, rmse = mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred, squared=False)
    plt = test_vs_pred_plot(y_test,y_pred, 'K-Neighbors Regressor Accuracy')
    return kn, mse, rs, rmse, plt

def runLinearRegression(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse, rs, rmse = mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred, squared=False)
    plt = test_vs_pred_plot(y_test,y_pred, 'Linear Regressor Accuracy')

    return lr, mse, rs, rmse, plt

if __name__ == "__main__":
    print('Just a utility file, nothing to see here :)')