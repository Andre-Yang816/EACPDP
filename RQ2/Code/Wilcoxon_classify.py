# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def read_data(dataset_path, metric):
    functions = ['CBSplus', 'BF', 'PF', 'KF', 'DFAC', 'TCA', 'BDA', 'JDA', 'JPDA','TNB']
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

    baseline_data = datas['CBSplus']
    metric_datas.append(baseline_data)

    for key, value in datas.items():
        if key == 'CBSplus':
            continue
        metric_datas.append(value)
        functions.append(key)

    return metric_datas, functions

def wilcoxon(l1, l2):
    w, p_value = stats.wilcoxon(l1, l2, correction=False)
    return p_value

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
    imp = round((avgl1-avgl2), 3)

    return imp

def Wilcoxon_signed_classify_test(metric_datas, functions, metric):
    pvalues = []
    sortpvalues = []
    bhpvalues = []
    print('***********{0}***********'.format(metric))
    for i in range(1, len(metric_datas)):
        pvalue = wilcoxon(metric_datas[0], metric_datas[i])
        pvalues.append(pvalue)
        sortpvalues.append(pvalue)
        print('------------{0}-----------------'.format(functions[i-1]))
        print("compute p-value between CBSplus and %s: %s" % (functions[i-1], pvalue))
        print("compute W/D/L between CBSplus and %s: %s" % (functions[i-1], wdl(metric_datas[0], metric_datas[i])))
        print("compute average improvement between CBSplus and %s: %s" % (functions[i-1],
                                                                             average_improvement(metric_datas[0], metric_datas[i])))

    sortpvalues.sort()

    for i in range(len(pvalues)):
        bhpvalue = pvalues[i]*(len(pvalues))/(sortpvalues.index(pvalues[i])+1)
        bhpvalues.append(bhpvalue)
        print("compute Benjamini—Hochberg p-value between %s and CBSplus: %s" % (functions[i-1], bhpvalue))
    Path('../classify/Testdata/{0}'.format(dataset)).mkdir(parents=True, exist_ok=True)
    output_path = '../classify/Testdata/%s/%s.csv' % (dataset, metric)
    output = pd.DataFrame(data=[bhpvalues], columns=functions)
    output.to_csv(output_path, encoding='utf-8')


if __name__ == '__main__':
    metrics = ['Precision', 'Recall', 'F1']
    result_path = '../../output_classify'
    for dataset in os.listdir(result_path):
        dataset_path = result_path + '/%s' % dataset
        for metric in metrics:
            print("Doing Wilcoxon signed classify test in %s_%s ..." % (dataset, metric))
            datas = read_data(dataset_path, metric)
            metric_datas, functions = process_data(datas)
            Wilcoxon_signed_classify_test(metric_datas, functions, metric)