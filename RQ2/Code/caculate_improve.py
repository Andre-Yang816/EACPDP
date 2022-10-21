import os
from pathlib import Path
import pandas as pd
import numpy as np

def read_data(dataset_path, metric):
    functions = ['None3', 'BF3', 'PF3', 'KF3', 'DFAC3', 'TCA3', 'BDA3', 'JDA3', 'JPDA3','TNB3']
    datas = {}
    for function in functions:
        data_path = '{0}/{1}.csv'.format(dataset_path,function)
        raw_datas = pd.read_csv(data_path)

        raw_datas = raw_datas[metric].values

        datas[function] = raw_datas

    return datas

def process_data(datas):
    metric_datas = []
    functions = []

    baseline_data = datas['None3']
    metric_datas.append(baseline_data)

    for key, value in datas.items():
        if key == 'None3':
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
    metrics = ['Precision', 'Recall', 'F1', 'PofB', 'PMI', 'Popt', 'IFA']
    result_path = '../../output_rank'
    for dataset in os.listdir(result_path):
        if dataset in ['KNN', 'LR', 'RF']:
            dataset_path = result_path + '/%s' % dataset
            for metric in metrics:
                datas = read_data(dataset_path, metric)
                metric_datas, functions = process_data(datas)
                imp_ave_dataset = caculate_improvement(metric_datas)
                Path('../Impdata/rank/{0}'.format(dataset)).mkdir(parents=True, exist_ok=True)
                output_path = '../Impdata/rank/%s/%s.csv' % (dataset, metric)
                output = pd.DataFrame(data=[imp_ave_dataset], columns=functions)
                output.to_csv(output_path, encoding='utf-8',index=False)

