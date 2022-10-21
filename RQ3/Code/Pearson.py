import copy
import math
import os
import pandas as pd
import numpy as np

def ReadFile(method,func):
    classify_path= '../../output_classify/{0}/{1}.csv'.format(method,func)
    rank_path = '../../output_rank/{0}/{1}3.csv'.format(method,func)
    data1 = pd.read_csv(classify_path)
    data1= data1[['Precision','Recall','F1']]
    data2 = pd.read_csv(rank_path)
    data = pd.concat([data1,data2], axis=1, join='outer', ignore_index=True)
    data_array = np.array(data)
    return data_array

def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean

def sckendall(a, b):
    L = len(a)
    count = 0
    for i in range(L - 1):
        for j in range(i + 1, L):
            count = count + np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    kendall_tau = count / (L * (L - 1) / 2)

    return kendall_tau

def calcPearson(x,y):
    x_mean,y_mean = calcMean(x,y)
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p

if __name__ == '__main__':
    functions=['BDA','JDA']
    methods = ['KNN', 'LR', 'RF']
    for func in functions:
        for method in methods:
            data_list = ReadFile(method,func)
            # data_new = copy.deepcopy(data_list)
            result_list = []
            for i in range(len(data_list[0])):
                x = list(data_list.T[i])
                tmp = []
                for j in range(len(data_list[0])):
                    y = list(data_list.T[j])
                    # tmp.append(calcPearson(x, y))
                    tmp.append(sckendall(x, y))
                result_list.append(tmp)

            print(result_list)
            index = ['Precision', 'Recall', 'F1',
                     'Precision@20%', 'Recall@20%', 'F1@20%', 'PofB@20%', 'PMI@20%', 'Popt', 'IFA']

            df = pd.DataFrame(data=result_list, columns=index)

            if not os.path.exists("../output_kendell/".format(func)):
                os.makedirs("../output_kendell/".format(func))
            df.to_csv("../output_kendell/{0}_{1}.csv".format(func,method), index=False)




