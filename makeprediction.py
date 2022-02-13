import sys
import pickle
import time

sys.path.insert(1, 'Utils/')
sys.path.insert(1, 'Processes/')

from dataProcessor import normalize
from regressorsUtils import *
from predictionUtils import *
from dataProcessingUtils import split_data


def make_prediction(quer):
    start_timer = time.time()
    date = find_chars_until_space(str(datetime.datetime.now()))
    csv_path = 'Datasets/house_data_{}.csv'.format(find_chars_until_space(str(datetime.datetime.now())))
    with open('../Models/model_{}'.format(date), 'wb') as f:
       model = pickle.load(f)
    
    clean_data, price_scaler = convert_query(csv_path, quer)
    prediction_scaled = model.predict(clean_data)
    prediction = reverse_scale(price_scaler, prediction_scaled)
    prediction_clean = round(prediction)
    end_timer = time.time()
    print('Prediction is {} and was made in {} seconds'.format(str(prediction_clean),str(round((end_timer - start_timer), 3))))
    return prediction_clean


if __name__ == "__main__":
    make_prediction(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])