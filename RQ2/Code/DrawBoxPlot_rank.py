import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def processDatas(datas):
    #baseline_data = 'CBSplus'
    #data_filter = [ 'BF', 'PF', 'KF', 'DFAC', 'TCA', 'BDA', 'JDA', 'JPDA']
    baseline_data = 'None4'
    data_filter = ['BF4', 'PF4', 'KF4', 'DFAC4','TCA4', 'BDA4', 'JDA4', 'JPDA4','TNB4']
    #data_filter = ['BDA3', 'JDA3', 'JPDA3', 'TNB3']
    metric_datas = []
    functions = []
    for function in data_filter:
        metric_datas.append(datas[function] - datas[baseline_data])
        functions.append(function)

    return metric_datas, functions

def load_color(dataset, metric, functions):
    colors_path = '../rank/Testdata/%s/%s.csv' % (dataset, metric)
    datas = pd.read_csv(colors_path)

    colors = []
    for function in functions:
        if datas[function][0] < 0.05:
            colors.append('red')
        else:
            colors.append('black')

    return colors

def drawFigure(metric_datas, functions, metric, dataset):
    ymax = 0
    ymin = 100
    for data in metric_datas:
        if ymax < max(data):
            ymax = max(data)
        if ymin > min(data):
            ymin = min(data)

    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(figsize=(4, 3))  # figsize:指定figure的宽和高，单位为英寸；
    ax.tick_params(direction='in')

    xticks = np.arange(1, len(functions) * 1.5, 1.5)
    figure = ax.boxplot(metric_datas,
                        notch=False,  # notch shape
                        sym='r+',  # blue squares for outliers
                        vert=True,  # vertical box aligmnent
                        meanline=True,
                        showmeans=False,
                        patch_artist=False,
                        showfliers=False,
                        positions=xticks,
                        boxprops={'color': 'red'}
                        )

    colors = load_color(dataset, metric, functions)
    for i in range(len(colors)):
        k = figure['boxes'][i]
        k.set(color=colors[i])
        k = figure['medians'][i]
        k.set(color=colors[i], linewidth=2)
        k = figure['whiskers'][2 * i:2 * i + 2]
        for w in k:
            w.set(color=colors[i], linestyle='--')
        k = figure['caps'][2 * i:2 * i + 2]
        for w in k:
            w.set(color=colors[i])

    plt.xlim((0, 14))
    functions_new = []
    for func in functions:
        functions_new.append(func[:-1])
    plt.xticks(xticks, functions_new, rotation=45, weight='heavy', fontsize=12, ha='center')
    plt.yticks(fontsize=12,weight='heavy')
    if metric not in ['IFA','Popt']:
        plt.ylabel(metric+'@20%', fontsize=12,weight='heavy')
    else:
        plt.ylabel(metric, fontsize=12,weight='heavy')
    plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

    plt.axhline(y=0, color='blue')

    plt.axvline(6.3, color='grey', linestyle=':')  # 添加一个竖线，4.5表示竖线的x轴坐标
    '''
    plt.axvline(18, color='grey', linestyle=':')  # 添加一个竖线，4.5表示竖线的x轴坐标
    plt.axvline(34, color='grey', linestyle=':')  # 添加一个竖线，4.5表示竖线的x轴坐标
    # plt.axvline(36.5, color='black', linestyle=':')  # 添加一个竖线，4.5表示竖线的x轴坐标
    '''
    plt.title(
        "     Data filter    "
        "    Transfer learning"
        , fontsize=14, loc='left', weight='heavy')
    Path('../rank/figures/{0}'.format(dataset)).mkdir(parents=True, exist_ok=True)
    output_path = '../rank/figures/%s/%s.jpg' % (dataset, metric)
    foo_fig = plt.gcf()
    foo_fig.savefig(output_path, format='jpg', dpi=1000, bbox_inches='tight')
    plt.clf()
    plt.close()

def read_data(dataset_path, metric):
    functions = ['None4', 'BF4', 'PF4', 'KF4', 'DFAC4', 'TCA4', 'BDA4', 'JDA4', 'JPDA4','TNB4']
    datas = {}
    for function in functions:
        data_path = '{0}/{1}.csv'.format(dataset_path,function)
        raw_datas = pd.read_csv(data_path)
        raw_datas = raw_datas[metric].values
        datas[function] = raw_datas
    return datas

def save_data(name,datas,classify):
    Path('../rank/Data/Prob/{0}'.format(classify)).mkdir(parents=True, exist_ok=True)
    output_path = '../rank/Data/Prob/{0}/{1}.csv'.format(classify,name)

    output = pd.DataFrame(data=datas)
    output.to_csv(output_path, encoding='utf-8')

def BoxPlot(results_path,classify):
    metrics = ['Precision', 'Recall', 'F1', 'PofB', 'PMI', 'Popt', 'IFA']
    #metrics = ['IFA']
    for dataset in os.listdir(results_path):
        if dataset == classify:
            for metric in metrics:
                dataset_path = results_path + '%s' % dataset
                datas = read_data(dataset_path, metric)
                save_data(metric, datas,classify)

if __name__ == '__main__':
    classifies = ['KNN', 'LR', 'RF']
    #classifies = ['KNN']
    for classify in classifies:
        BoxPlot(r'../output_rank/',classify)

