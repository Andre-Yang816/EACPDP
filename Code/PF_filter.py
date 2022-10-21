import time
import pandas as pd
from sklearn.cluster import KMeans

from Code.Processing import get_distance
from Code.PerformanceMeasure import *

def PF_filter(classifier,train_data_x, train_data_y, test_data_x, test_data_y):
    flag = [0] * len(train_data_x)

    train_data_len = len(train_data_x)
    test_data_len = len(test_data_x)

    data_len = train_data_len + test_data_len
    cluster_num = int(data_len / 30)

    all_data = np.concatenate((train_data_x,test_data_x),axis=0)
    kmeans_model = KMeans(n_clusters=cluster_num)
    kmeans_model.fit(all_data)

    cluster_label = kmeans_model.labels_
    cluster_flag = [0] * cluster_num

    cluster_index = []
    for i in range(cluster_num):
        cluster_index.append([])
    for i in range(data_len):
        cluster_index[cluster_label[i]].append(i)


    for i in range(test_data_len):
        cluster_flag[cluster_label[i+train_data_len]] = 1

    for i in range(train_data_len):
        if cluster_flag[cluster_label[i]] == 0:
            flag[i] = -1

    test_index = []
    for i in range(train_data_len):
        if flag[i] != -1:
            distance_set = []
            for j in range(len(cluster_index[cluster_label[i]])):
                tmp_index = cluster_index[cluster_label[i]][j]
                if tmp_index < train_data_len:
                    continue
                else:
                    distance_set.append(get_distance(all_data[i],all_data[tmp_index]))
            min_index = distance_set.index(min(distance_set))
            if min_index not in test_index:
                test_index.append(min_index)

    for i in range(len(test_index)):
        distance_set = []
        for j in range(len(cluster_index[cluster_label[i]])):
            tmp_index = cluster_index[cluster_label[i]][j]
            if tmp_index >= train_data_len:
                continue
            else:
                distance_set.append(get_distance(all_data[tmp_index],all_data[test_index[i]]))
        min_index = distance_set.index(min(distance_set))
        flag[min_index] = 1

    train_index = []
    for i in range(len(flag)):
        if flag[i] == 1:
            train_index.append(i)
    train_data_x = train_data_x[train_index]
    train_data_y = train_data_y[train_index]
    train_data_x = pd.DataFrame(train_data_x)
    train_data_y = pd.DataFrame(train_data_y)

    model = classifier(train_data_x, train_data_y)

    return model
