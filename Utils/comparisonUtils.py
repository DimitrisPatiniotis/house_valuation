import numpy as np

def find_max_avg(list_of_lists):
    max_avg = -100
    selected_list = []
    for i in list_of_lists:
        if np.average(i) > max_avg:
            selected_list = i
            max_avg = np.average(i)
    return selected_list, max_avg



if __name__ == "__main__":
   print('Just a utility file, nothing to see here :)')