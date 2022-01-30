import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Regression Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Scalers
from sklearn.preprocessing import MinMaxScaler

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, r2_score

housedata = pd.read_csv("house_data_2022-01-26.csv")
housedata['lvl'] = housedata['lvl'].replace('Υπερυψωμένο', 0.5).replace('Υπόγειο', -1).replace('Ημιώροφ', 0.5)

dictionary =  {
    'α' : 'a',
    'Α' : 'a',
    'ά' : 'a',
    'Ά' : 'a',
    'β' : 'b',
    'Β' : 'b',
    'γ' : 'g',
    'Γ' : 'g',
    'δ' : 'd',
    'Δ' : 'd',
    'ε' : 'e',
    'έ' : 'e',
    'Ε' : 'e',
    'Έ' : 'e',
    'ζ' : 'z',
    'Ζ' : 'z',
    'η' : 'h',
    'ή' : 'h',
    'Η' : 'h',
    'Ή' : 'h',
    'θ' : 'th',
    'Θ' : 'th',
    'ι' : 'i',
    'ί' : 'i',
    'ϊ' : 'i',
    'Ι' : 'i',
    'Ί' : 'i',
    'κ' : 'k',
    'Κ' : 'k',
    'λ' : 'l',
    'Λ' : 'l',
    'μ' : 'm',
    'Μ' : 'm',
    'ν' : 'n',
    'Ν' : 'n',
    'ξ' : 'x',
    'Ξ' : 'x',
    'ο' : 'o',
    'ό' : 'o',
    'Ο' : 'o',
    'Ό' : 'o',
    'π' : 'p',
    'Π' : 'p',
    'ρ' : 'r',
    'Ρ' : 'r',
    'σ' : 's',
    'Σ' : 's',
    'ς' : 's',
    'τ' : 't',
    'Τ' : 't',
    'υ' : 'u',
    'ύ' : 'u',
    'Υ' : 'u',
    'Ύ' : 'u',
    'φ' : 'f',
    'Φ' : 'f',
    'χ' : 'x',
    'Χ' : 'x',
    'ψ' : 'ps',
    'Ψ' : 'ps',
    'ω' : 'w',
    'ώ' : 'w',
    'Ω' : 'w',
    'Ώ' : 'w',
    ' ' : ' ',
}

def gr_to_en(grstr):
    return ''.join([dictionary.get(i) for i in grstr])

housedata['loc'] = housedata['loc'].apply(lambda x: gr_to_en(x))
housedata['type'] = housedata['type'].apply(lambda x : gr_to_en(x))

print(len(housedata))
# Droping outliers
upper_lim = housedata['price'].quantile(.999)
lower_lim = housedata['price'].quantile(.001)
# housedata = housedata[(housedata['price'] < upper_lim) & (housedata['price'] > lower_lim)]
print(len(housedata))


def label_encoding(columns_list, dataframe):
    for col in columns_list:
        dataframe[col] = dataframe[col].astype('category')
        dataframe[col] = dataframe[col].cat.codes
    return dataframe

def one_hot_encoding(columns_list, dataframe):
    for col in columns_list:
        column = pd.get_dummies(dataframe[col])
        dataframe = dataframe.join(column).drop(col, axis=1)
    return dataframe

def min_max_scaler(columns_list, dataframe):
    scaler = MinMaxScaler()
    for col in columns_list:
        dataframe[col] = scaler.fit_transform(dataframe[col].array.reshape(-1,1))
    return dataframe

def min_max_inverse(columns_list, dataframe):
    scaler = MinMaxScaler()
    for col in columns_list:
        dataframe[col] = scaler.inverse_transform(dataframe[col].array.reshape(-1,1))
    return dataframe

def linReg(X_train, X_test, y_train, y_test):
    linReg = LinearRegression()
    linReg.fit(X_train, y_train)

    y_pred = linReg.predict(X_test)
    plt.scatter(y_test, y_pred,color='g')
    plt.show()
    print("Linear Regression mean squared error: %.5f" % mean_squared_error(y_test, y_pred))
    

def supportVectorReg(X_train, X_test, y_train, y_test):

    svr_regressor = SVR(kernel='rbf')
    svr_regressor.fit(X_train, y_train)

    y_pred = svr_regressor.predict(X_test)
    plt.scatter(y_test, y_pred,color='g')
    plt.show()
    print("Support Vector Regression mean squared error: %.5f" % mean_squared_error(y_test, y_pred))



housedata = one_hot_encoding(['loc', 'type'], housedata)
housedata = min_max_scaler(['price', 'year'], housedata)
# print(housedata.head(5))

X = housedata.drop('price', axis=1)
y = housedata['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

linReg(X_train, X_test, y_train, y_test)
supportVectorReg(X_train, X_test, y_train, y_test)
