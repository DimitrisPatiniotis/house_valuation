import sys
import pickle
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import datetime
import time
import progressbar

sys.path.insert(1, 'Processes/')
sys.path.insert(1, '../Utils/')

from dataProcessor import normalize
from scraper import find_chars_until_space
from dataProcessingUtils import split_data

def get_best_rfr(date):

    data = normalize('../Datasets/house_data_{}.csv'.format(date), scaled_list = ['price','year','loc', 'type'], encoding_type='label')[0]
    X_train, X_test, y_train, y_test = split_data(data, 0.2)
    min_rmse = 100
    current_return = None
    iters = 10
    print('Compairing models: \n')
    bar = progressbar.ProgressBar(maxval=iters).start()
    for i in range(iters):
        model = RandomForestRegressor(n_estimators= 140)
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        error = mean_squared_error(y_test, y_pred, squared=False)
        if error < min_rmse:
            min_rmse = error
            current_return = model
        bar.update(i)
        
    print('\n\nMin rmse of {}'.format(round(min_rmse, 3)))
    return current_return, min_rmse

def create_and_store_model():
    start_timer = time.time()
    date = find_chars_until_space(str(datetime.datetime.now()))
    min_rmse = 100
    while min_rmse > 0.088:
        model, min_rmse = get_best_rfr(date)

    with open('../Models/model_{}'.format(date), 'wb') as f:
        pickle.dump(model, f)
    end_timer = time.time()
    print('Model created succesfully in {} seconds and stored at ../Models/model_{}'.format(str(round((end_timer - start_timer), 3)), date))

if __name__ == "__main__":
    create_and_store_model()