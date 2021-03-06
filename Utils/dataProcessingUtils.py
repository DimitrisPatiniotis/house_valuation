# General
import pandas as pd
import datetime
from matplotlib import pyplot as plt

# Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Train Test Split
from sklearn.model_selection import train_test_split

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

def reverse_year(year):
    return  datetime.now().year - int(year)

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

def min_max_scaler(col, dataframe):
    scaler = MinMaxScaler()
    dataframe[col] = scaler.fit_transform(dataframe[col].array.reshape(-1,1))
    return dataframe, scaler

def standard_scaler(col, dataframe):
    scaler = StandardScaler()
    dataframe[col] = scaler.fit_transform(dataframe[col].array.reshape(-1,1))
    return dataframe, scaler

def split_data(data, test_size):
    X = data.drop('price', axis=1)
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def display_price_destr(price_col):
    plt.xlabel('Price')
    plt.ylabel('Number of samples')
    plt.title('Price Distribution')
    plt.plot()
    plt.hist(price_col, facecolor = 'blue', edgecolor='peru', bins=15)
    plt.show()

if __name__ == "__main__":
   print('Just a utility file, nothing to see here :)')