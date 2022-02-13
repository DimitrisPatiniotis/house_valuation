import pandas as pd
import sys
sys.path.insert(1, '../Processes/')

from dataProcessor import *

def list_to_df(input_list):
    return pd.DataFrame([input_list], columns=['type','loc','sqm','lvl','nbed','nbath','year', 'price']) 

def convert_query(csv_path, query_params=['Διαμέρισμα', 'Καμίνια', 78, 3, 1 ,1, 2001]):
    try:
        dataset = pd.read_csv(csv_path)
    except:
        print('Please enter a valid csv path')

    query_params.append(round((dataset['price'].mean())))

    query_unscaled = list_to_df(query_params)

    scaled_frame, scalers = normalize(csv_file_path = csv_path, added_row = query_unscaled, scaled_list = ['price','year','loc', 'type'],encoding_type='label')

    scaled_query = scaled_frame.tail(1).drop('price', axis=1)

    return scaled_query, scalers[0]

def reverse_scale(scaler, price):
    return scaler.inverse_transform(price)

if __name__ == "__main__":
   convert_query(csv_path='../Datasets/house_data_2022-01-26.csv')