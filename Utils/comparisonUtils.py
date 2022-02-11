import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# Regression Algorithms
from sklearn.linear_model import LinearRegression, BayesianRidge, SGDRegressor, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

# Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, r2_score




def min_max_scaler(columns_list, dataframe):
    for col in columns_list:
        scaler = MinMaxScaler()
        dataframe[col] = scaler.fit_transform(dataframe[col].array.reshape(-1,1))
    return dataframe, scaler

def min_max_inverse(columns_list, dataframe, scaler):
    for col in columns_list:
        dataframe[col] = scaler.inverse_transform(dataframe[col].array.reshape(-1,1))
    return dataframe

def linReg(X_train, X_test, y_train, y_test):
    linReg = LinearRegression()
    linReg.fit(X_train, y_train)

    y_pred = linReg.predict(X_test)
    # plt.scatter(y_test, y_pred,color='g')
    # plt.show()
    print("Linear Regression mean squared error: %.5f" % mean_squared_error(y_test, y_pred))
    return linReg

def supportVectorReg(X_train, X_test, y_train, y_test):

    svr_regressor = SVR(kernel='rbf')
    svr_regressor.fit(X_train, y_train)

    y_pred = svr_regressor.predict(X_test)
    # plt.scatter(y_test, y_pred,color='g')
    # plt.show()
    print("Support Vector Regression mean squared error: %.5f" % mean_squared_error(y_test, y_pred))

def bayesianR(X_train, X_test, y_train, y_test):
    clf = BayesianRidge(compute_score=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Bayesian Ridge Regression mean squared error: %.5f" % mean_squared_error(y_test, y_pred))

def kNeighbor(X_train, X_test, y_train, y_test):
    knn = KNeighborsRegressor()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("kNN Regression mean squared error: %.5f" % mean_squared_error(y_test, y_pred))

def randomForest(X_train, X_test, y_train, y_test):
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Random Forest Regression mean squared error: %.5f" % mean_squared_error(y_test, y_pred))
    return rf, mean_squared_error(y_test, y_pred), 

def lassoRegressor(X_train, X_test, y_train, y_test):
    las = Lasso(alpha = 0.1)
    las.fit(X_train, y_train)
    y_pred = las.predict(X_test)
    print("Lasso Regression mean squared error: %.5f" % mean_squared_error(y_test, y_pred))
    return y_pred, las

# print(housedata.iloc[-1]['price'])
housedata, mm_scaler = min_max_scaler(['price', 'year'], housedata)
last_price = housedata.iloc[-1]['price']
# print(type(last_price))

# sns.pairplot(housedata)
# plt.show()



last_row = housedata.tail(1).drop('price', axis=1)
housedata.drop(housedata.tail(1).index,inplace=True)
# print(last_row)

X = housedata.drop('price', axis=1)

y = housedata['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def main():
    pass

if __name__ == "__main__":

   main()