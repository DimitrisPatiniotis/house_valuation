from urllib import response
from django.shortcuts import render
import datetime
import re

from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


import sys
import pickle
import time

def find_chars_until_space(str):
    find = re.compile(r"^[^ ]*")
    m = re.search(find, str).group(0)
    return m

def normalize(csv_file_path, outlier_field = 'price', outlier_upper = .99, outlier_lower = .01, encoding_type = 'one_hot', encoding_list = ['loc', 'type'], scaling_method = 'min_max', scaled_list = ['price','year'], added_row = None):

    print('Starting data processing')

    start_timer = time.time()

    try:
        housedata = pd.read_csv(csv_file_path)
    except:
        print('Please enter a valid csv path')
    housedata = housedata[housedata.nbed < 10]
    housedata = housedata[housedata.nbath < 10]
    # # housedata = housedata[housedata.lvl < 8]
    housedata = housedata[housedata.sqm < 500]
    # print(housedata['sqm'].max())
    # # Droping outliers
    try:
        upper_lim = housedata[outlier_field].quantile(outlier_upper)
        lower_lim = housedata[outlier_field].quantile(outlier_lower)
        housedata = housedata[(housedata[outlier_field] < upper_lim) & (housedata['price'] > lower_lim)]
    except:
        print('Please insert valid outlier limits (range .00 to .99) and a valid field name')
        return False

    if isinstance(added_row, pd.DataFrame):
        housedata = pd.concat([housedata, added_row], ignore_index = True, axis = 0)



    # Replacing string instances from lvl column
    housedata['lvl'] = housedata['lvl'].replace('Υπερυψωμένο', 0.5).replace('Υπόγειο', -1).replace('Ημιώροφ', 0.5)

    housedata['loc'] = housedata['loc'].apply(lambda x: gr_to_en(x))
    housedata['type'] = housedata['type'].apply(lambda x : gr_to_en(x))

    # Encoding
    if encoding_type == 'one_hot':
        try:
            housedata = one_hot_encoding(encoding_list, housedata)
        except:
            print('Please enter a valid list of columns you want to be encoded')
    elif encoding_type == 'label':
        try:
            housedata = label_encoding(encoding_list, housedata)
        except:
            print('Please enter a valid list of columns you want to be encoded')
    else:
        print('Please chose between \'one_hot\' and \'label\' as the encoding method')

    # Data Scaling
    scalers = []
    if scaling_method == 'min_max':
        for i in scaled_list:
            housedata, locals()[str(i) + '_scaler'] = min_max_scaler(i, housedata)
            if i == 'price': scalers.append(locals()[str(i) + '_scaler'])
    elif scaling_method == 'standard':
        for i in scaled_list:
            housedata, locals()[str(i) + '_scaler'] = standard_scaler(i, housedata)
            if i == 'price': scalers.append(locals()[str(i) + '_scaler'])
    elif scaling_method == 'log':
        for i in scaled_list:
            transformer = FunctionTransformer(np.log1p, validate=True)
            transformer.transform(housedata[i].array.reshape(-1,1))
            display_price_destr(housedata['price'])
            if i == 'price': scalers.append(transformer)
    
    end_timer = time.time()
    print('Data processed succesfully in {} seconds'.format(str(round((end_timer - start_timer), 3))))
    return housedata, scalers
    
def list_to_df(input_list):
    return pd.DataFrame([input_list], columns=['type','loc','sqm','lvl','nbed','nbath','year', 'price']) 

def convert_query(csv_path, query_params):
    try:
        dataset = pd.read_csv(csv_path)
    except:
        print('Please enter a valid csv path')

    query_params.append(round((dataset['price'].mean())))

    query_unscaled = list_to_df(query_params)

    scaled_frame, scalers = normalize(csv_file_path = csv_path, added_row = query_unscaled, scaled_list = ['price','year','loc', 'type', 'sqm', 'lvl', 'nbed','nbath'],encoding_type='label')

    scaled_query = scaled_frame.tail(1).drop('price', axis=1)

    return scaled_query, scalers[0]

def reverse_scale(scaler, price):
    return scaler.inverse_transform(price)

def make_prediction(quer):
    start_timer = time.time()
    date = find_chars_until_space(str(datetime.datetime.now()))
    csv_path = 'Datasets/house_data_{}.csv'.format(find_chars_until_space(str(datetime.datetime.now())))
    with open('../Models/model_{}.pickle'.format(date), 'rb') as f:
        unpickler = pickle.Unpickler(f)
        model = unpickler.load()
    
    clean_data, price_scaler = convert_query(csv_path, quer)
    prediction_scaled = model.predict(clean_data)
    prediction = reverse_scale(price_scaler, prediction_scaled.reshape(1,-1))
    prediction_clean = round(prediction[0][0])
    end_timer = time.time()
    print('The prediction is {}€ and was made in {} seconds'.format(str(prediction_clean),str(round((end_timer - start_timer), 3))))
    return prediction_clean



class call_model(APIView):

    def get(self,request):
        if request.method == 'GET':
            
            # sentence is the query we want to get the prediction for
            params =  request.GET.get('list')
            
            # predict method used to get the prediction
            response = make_prediction(params)
            
            # returning JSON response
            return JsonResponse(response)