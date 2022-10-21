# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def read_data(dataset_path, metric):
    functions = ['None3', 'BF3', 'PF3', 'KF3', 'DFAC3', 'TCA3', 'BDA3', 'JDA3', 'JPDA3','TNB3']
    #functions = ['None1', 'BF1', 'PF1', 'KF1', 'DFAC1', 'TCA1', 'BDA1', 'JDA1', 'JPDA1']
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

def wilcoxon(l1, l2):
    w, p_value = stats.wilcoxon(l1, l2, correction=False)
    return p_value

# win draw loss值
def wdl(l1, l2):
    win = 0
    draw = 0
    loss = 0
    for i in range(len(l1)):
        if l1[i] < l2[i]:
            loss = loss+1
        if l1[i] == l2[i]:
            draw = draw+1
        if l1[i] > l2[i]:
            win = win+1

    return win, draw, loss

def average_improvement(l1, l2):
    avgl1 = round(np.average(l1), 3)
    avgl2 = round(np.average(l2), 3)
    #imp = round((avgl1-avgl2)/avgl2, 4)
    imp = round((avgl1-avgl2), 3)
    return imp

def Wilcoxon_signed_rank_test(metric_datas, functions, metric):
    pvalues = []
    sortpvalues = []
    bhpvalues = []
    print('***********{0}***********'.format(metric))

    improve_ave_dataset = []
    for i in range(1, len(metric_datas)):

        pvalue = wilcoxon(metric_datas[0], metric_datas[i])
        pvalues.append(pvalue)
        sortpvalues.append(pvalue)
        improve_ave_dataset.append(average_improvement(metric_datas[i], metric_datas[0]))
        if metric == 'Recall' or metric == 'PofB':
            print('-------------{0}-----------------'.format(functions[i - 1]))
            #print("compute p-value between %s and CBSplus: %s" % (functions[i-1], pvalue))
            #print("compute W/D/L between %s and CBSplus: %s" % (functions[i-1], wdl(metric_datas[i], metric_datas[0])))
            print("compute average improvement between {0} and CBSplus: {1}" .format(functions[i-1],
                                                                             average_improvement(metric_datas[i], metric_datas[0])))

    sortpvalues.sort()

    for i in range(len(pvalues)):
        bhpvalue = pvalues[i]*(len(pvalues))/(sortpvalues.index(pvalues[i])+1)
        bhpvalues.append(bhpvalue)
        print("compute Benjamini—Hochberg p-value between %s and CBSplus: %s" % (functions[i-1], bhpvalue))

    Path('../rank/Testdata/{0}'.format(dataset)).mkdir(parents=True, exist_ok=True)
    output_path = '../rank/Testdata/%s/%s.csv' % (dataset, metric)

    output = pd.DataFrame(data=[bhpvalues], columns=functions)
    #output = pd.DataFrame(data=[pvalues], columns=functions)
    output.to_csv(output_path, encoding='utf-8')
    return improve_ave_dataset


if __name__ == '__main__':
    metrics = ['Precision', 'Recall', 'F1', 'PofB', 'PMI', 'Popt', 'IFA']
    result_path = '../../output_rank'
    for dataset in os.listdir(result_path):
        #if dataset =='KNN':
        dataset_path = result_path + '/%s' % dataset
        for metric in metrics:
            print("Doing Wilcoxon signed rank test in %s_%s ..." % (dataset, metric))
            datas = read_data(dataset_path, metric)
            metric_datas, functions = process_data(datas)
            imp_ave_dataset = Wilcoxon_signed_rank_test(metric_datas, functions, metric)

            Path('../Impdata/rank/{0}'.format(dataset)).mkdir(parents=True, exist_ok=True)
            output_path = '../Impdata/rank/%s/%s.csv' % (dataset, metric)

            output = pd.DataFrame(data=[imp_ave_dataset], columns=functions)
            output.to_csv(output_path, encoding='utf-8')