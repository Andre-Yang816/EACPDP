from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def readfile(classify,func):
    path='../output/median/{0}/l0.'.format(classify)
    dataset = pd.read_csv('{0}1/{1}.csv'.format(path,func),index_col=False)
    for i in range(2,10):
        filepath = path + str(i) + '/{0}.CSV'.format(func)
        raw_datas = pd.read_csv(filepath)
        dataset = pd.concat([dataset,raw_datas])
    dataset = dataset.loc[:,'Precision':'IFA']
    return dataset


def draw_pic(dataset,func):
    x = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    Precision = dataset['Precision'].tolist()
    Recall = dataset['Recall'].tolist()
    F1 = dataset['F1'].tolist()
    PofB = dataset['PofB'].tolist()
    PMI = dataset['PMI'].tolist()
    Popt = dataset['Popt'].tolist()
    IFA = dataset['IFA'].tolist()
    plt.plot(x, Precision, color='tomato', marker='o', linestyle='-', label='Precision@20%')
    plt.plot(x, Recall, color='sandybrown', marker='D', linestyle='-', label='Recall@20%')
    plt.plot(x, F1, color='orangered', marker='*', linestyle='-', label='F1@20%')
    plt.plot(x, PofB, color='skyblue', marker='+', linestyle='-.', label='PofB@20%')
    plt.plot(x, PMI, color='royalblue', marker='p', linestyle='-.', label='PMI@20%')
    plt.plot(x, Popt, color='slateblue', marker='v', linestyle='-.', label='Popt')
    plt.legend(bbox_to_anchor=(0.5, -0.3), loc=8, ncol=3)
    plt.xlabel(chr(955))

    Path('../Picture/median').mkdir(parents=True, exist_ok=True)
    output_path = '../Picture/median/{0}.jpg'.format(func)
    foo_fig = plt.gcf()
    foo_fig.savefig(output_path, format='jpg', dpi=1000, bbox_inches='tight')
    plt.clf()
    return IFA

def draw():
    x = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    classifiers = ['KNN', 'LR', 'RF']
    BDA_KNN = readfile(classifiers[0],'BDA3')
    JDA_KNN = readfile(classifiers[0], 'JDA3')
    BDA_IFA = draw_pic(BDA_KNN,'BDA')
    JDA_IFA = draw_pic(JDA_KNN,'JDA')
    plt.plot(x, BDA_IFA, color='sandybrown', marker='o', linestyle='-', label='BDA')
    plt.plot(x, JDA_IFA, color='skyblue', marker='D', linestyle='--', label='JDA')
    plt.legend(bbox_to_anchor=(0.5, -0.3), loc=8,ncol=2)
    plt.xlabel(chr(955))
    plt.ylabel('IFA', fontsize=14, family='Times New Roman', weight='heavy')
    Path('../Picture/median').mkdir(parents=True, exist_ok=True)
    output_path = '../Picture/median/IFA.jpg'
    foo_fig = plt.gcf()
    foo_fig.savefig(output_path, format='jpg', dpi=1000, bbox_inches='tight')
    plt.clf()
if __name__ == '__main__':
    draw()