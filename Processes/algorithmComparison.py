import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Regression Algorithms
from sklearn.linear_model import LinearRegression, BayesianRidge, SGDRegressor, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, r2_score

from utils import one_hot_encoding, min_max_scaler, reverse_year, dictionary, gr_to_en, randomForest, lassoRegressor, linReg

last_price_prev = housedata.iloc[-1]['price']
print(last_price_prev)
housedata, mm_scaler = min_max_scaler(['year','lvl', 'nbath'], housedata)
scaler = MinMaxScaler()
housedata['price'] = scaler.fit_transform(housedata['price'].array.reshape(-1,1))
last_price = housedata.iloc[-1]['price']
print(scaler.inverse_transform(last_price.reshape((1,-1))))
last_row = housedata.tail(1).drop('price', axis=1)
housedata.drop(housedata.tail(1).index,inplace=True)

print(housedata.head(5))


X = housedata.drop('price', axis=1)

y = housedata['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


rf = randomForest(X_train, X_test, y_train, y_test)
print(scaler.inverse_transform(rf.predict(last_row).reshape((1,-1))))


lr = linReg(X_train, X_test, y_train, y_test)
print(scaler.inverse_transform(lr.predict(last_row).reshape((1,-1))))