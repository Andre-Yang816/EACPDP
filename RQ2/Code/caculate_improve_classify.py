# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
def read_data(dataset_path, metric):
    functions = ['CBSplus', 'BF', 'PF', 'KF', 'DFAC', 'TCA', 'BDA', 'JDA', 'JPDA', 'TNB']
    datas = {}
    for function in functions:
        data_path = '{0}/{1}.csv'.format(dataset_path, function)
        raw_datas = pd.read_csv(data_path)
        raw_datas = raw_datas[metric].values
        datas[function] = raw_datas
    return datas


def process_data(datas):
    metric_datas = []
    functions = []
    baseline_data = datas['CBSplus']
    metric_datas.append(baseline_data)
    for key, value in datas.items():
        if key == 'CBSplus':
            continue
        metric_datas.append(value)
        functions.append(key)
    return metric_datas, functions

def caculate_improvement(metric_datas):
    improve_ave_dataset = []
    for i in range(1, len(metric_datas)):
        improve_ave_dataset.append(np.around(np.median(metric_datas[i] - metric_datas[0]),3))
    return improve_ave_dataset

if __name__ == '__main__':

    metrics = ['Precision', 'Recall', 'F1']
    result_path = '../../output_classify'
    for dataset in os.listdir(result_path):
        dataset_path = result_path + '/%s' % dataset
        for metric in metrics:
            datas = read_data(dataset_path, metric)
            metric_datas, functions = process_data(datas)
            imp_ave_dataset = caculate_improvement(metric_datas)
            Path('../Impdata/classify/{0}'.format(dataset)).mkdir(parents=True, exist_ok=True)
            output_path = '../Impdata/classify/%s/%s.csv' % (dataset, metric)
            output = pd.DataFrame(data=[imp_ave_dataset], columns=functions)
            output.to_csv(output_path, encoding='utf-8', index=False)
