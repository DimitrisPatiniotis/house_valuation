import sys
import pickle
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import datetime
import time
import progressbar

sys.path.insert(1, 'Processes/')
sys.path.insert(1, '../Utils/')

from dataProcessor import normalize
from scraper import find_chars_until_space
from dataProcessingUtils import split_data

def get_best_rfr(date):

    data = normalize('../Datasets/house_data_{}.csv'.format(date), scaled_list = ['price','year','loc', 'type', 'sqm', 'lvl'], encoding_type='label')[0]
    X_train, X_test, y_train, y_test = split_data(data, 0.2)

    max_r2 = 0
    current_return = None
    iters = 10
    print('Compairing models: \n')
    bar = progressbar.ProgressBar(maxval=iters).start()
    for i in range(iters):
        model = RandomForestRegressor(n_estimators= 140)
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        if r2 > max_r2:
            max_r2 = r2
            current_return = model
        bar.update(i)
        
    print('\n\nMin rmse of {}'.format(round(max_r2, 3)))
    return current_return, r2

def create_and_store_model():
    start_timer = time.time()
    date = find_chars_until_space(str(datetime.datetime.now()))
    r2 = 0
    while r2 < 0.7:
        model, r2 = get_best_rfr(date)
    with open('../Models/model_{}.pickle'.format(date), 'wb') as f:
        pickle.dump(model, f)
    end_timer = time.time()
    print('Model created succesfully in {} seconds and stored at ../Models/model_{}'.format(str(round((end_timer - start_timer), 3)), date))

if __name__ == "__main__":
    create_and_store_model()