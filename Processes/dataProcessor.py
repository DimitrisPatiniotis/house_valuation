from math import nan
import numpy as np
import time
import sys
from sklearn.preprocessing import FunctionTransformer
sys.path.insert(1, '../Utils/')

from dataProcessingUtils import *


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


if __name__ == "__main__":
   normalize('../Datasets/house_data_2022-01-26.csv')