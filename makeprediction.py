import sys


sys.path.insert(1, 'Utils/')
sys.path.insert(1, 'Processes/')

from dataProcessor import normalize
from regressorsUtils import *
from predictionUtils import *
from dataProcessingUtils import split_data


def make_prediction():
    pass


if __name__ == "__main__":
    make_prediction(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])