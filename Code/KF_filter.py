import time
import pandas as pd
from sklearn.cluster import DBSCAN
from Code.PerformanceMeasure import *
from Code.Processing import get_distance
def KF_filter(classifier,train_data_x, train_data_y, test_data_x, test_data_y):
    train_data_len = len(train_data_x)
    all_data = np.concatenate((train_data_x, test_data_x), axis=0)
    all_size = len(all_data)
    data_distance = np.zeros((all_size, all_size))
    for i in range(all_size):
        for j in range(all_size):
            if i == j:
                continue
            elif i > j:
                data_distance[i][j] = data_distance[j][i]
            else:
                data_distance[i][j] = get_distance(all_data[i], all_data[j])

    clusters = DBSCAN(min_samples=10).fit_predict(data_distance)
    cluster_num = len(set(clusters))
    final_instance = []
    for i in range(cluster_num):
        flag = False
        for j in range(test_data_x.shape[0]):
            if i == clusters[j]:
                flag = True
                break
        if flag == True:
            for p in range(0, train_data_len):
                if clusters[p] == i:
                    final_instance.append(p)
    train_data_x = train_data_x[final_instance]
    train_data_y = train_data_y[final_instance]
    train_data_x = pd.DataFrame(train_data_x)
    train_data_y = pd.DataFrame(train_data_y)
    model = classifier(train_data_x, train_data_y)


    return model
