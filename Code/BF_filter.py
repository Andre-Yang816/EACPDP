import time
import pandas as pd

from Code.Processing import get_distance
from Code.PerformanceMeasure import *

def BF_filter(classifier,train_data_x, train_data_y, test_data_x, test_data_y):
    print("BF_Filter start time：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    flag = [0] * len(train_data_x)
    for test_instance in test_data_x:
        distance_set = []
        for train_instance in train_data_x:
            distance = get_distance(train_instance.tolist(),test_instance.tolist())
            distance_set.append(distance)
        for i in range(10):
            min_index = distance_set.index(min(distance_set))
            distance_set[min_index] = math.inf
            flag[min_index] = 1
    train_index = []
    for i in range(len(flag)):
        if flag[i] == 1:
            train_index.append(i)
    train_data_x = train_data_x[train_index]
    train_data_y = train_data_y[train_index]
    train_data_x = pd.DataFrame(train_data_x)
    train_data_y = pd.DataFrame(train_data_y)
    train_data_y = train_data_y.astype('int')
    model = classifier(train_data_x, train_data_y)

    return model

