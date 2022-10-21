import pandas as pd
import numpy as np
import math

def split_data(dataset):
    dataset = np.array(dataset,dtype=float)
    k = len(dataset[0])

    y = dataset[:, k - 1]
    label = [1 if y[i] >= 1 else 0 for i in range(len(y))]
    label = np.array(label, dtype=int)
    tmp_x = dataset[:, 0:k - 1]
    loc = tmp_x[:, 10]
    x = np.delete(tmp_x, 10, axis=1)

    return x, label, loc

def split_data2(dataset):
    dataset = np.array(dataset,dtype=float)
    k = len(dataset[0])

    y = dataset[:, k - 1]
    y = np.array(y, dtype=int)
    tmp_x = dataset[:, 0:k - 1]
    loc = tmp_x[:, 10]
    x = np.delete(tmp_x, 10, axis=1)

    return x, y, loc

def get_distance(instance1, instance2):
    for i in range(len(instance1)):
        instance1[i] = math.log(instance1[i] + 1) / math.log(2)
        instance2[i] = math.log(instance2[i] + 1) / math.log(2)
    sum = 0
    for i in range(len(instance1)):
        sum += math.pow(instance1[i]-instance2[i], 2)

    distance = math.sqrt(sum)

    return distance
