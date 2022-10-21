import os
import pandas as pd

def read_data(dataset_path, metric):

    functions = ['None1', 'BF1', 'PF1', 'KF1', 'DFAC1', 'TCA1', 'BDA1', 'JDA1', 'JPDA1','TNB1',
                      'None2', 'BF2', 'PF2', 'KF2', 'DFAC2', 'TCA2', 'BDA2', 'JDA2', 'JPDA2','TNB2',
                      'None3', 'BF3', 'PF3', 'KF3', 'DFAC3', 'TCA3', 'BDA3', 'JDA3', 'JPDA3','TNB3',
                      'None4', 'BF4', 'PF4', 'KF4', 'DFAC4', 'TCA4', 'BDA4', 'JDA4', 'JPDA4','TNB4']
    datas = {}
    for function in functions:
        data_path = '{0}{1}.csv'.format(dataset_path,function)
        raw_datas = pd.read_csv(data_path)
        raw_datas = raw_datas[metric].values
        datas[function] = raw_datas
    return datas

def combineData(dataset,classify,metric):
    functions = ['None1', 'BF1', 'PF1', 'KF1', 'DFAC1', 'TCA1', 'BDA1', 'JDA1', 'JPDA1', 'TNB1',
                 'None2', 'BF2', 'PF2', 'KF2', 'DFAC2', 'TCA2', 'BDA2', 'JDA2', 'JPDA2', 'TNB2',
                 'None3', 'BF3', 'PF3', 'KF3', 'DFAC3', 'TCA3', 'BDA3', 'JDA3', 'JPDA3', 'TNB3',
                 'None4', 'BF4', 'PF4', 'KF4', 'DFAC4', 'TCA4', 'BDA4', 'JDA4', 'JPDA4', 'TNB4']
    result = []
    for func in functions:
        result.append(dataset[func])
    df = pd.DataFrame(data=result,index=functions)
    final = df.T
    print(final)
    write_path = '../target/{0}'.format(classify)
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    final.to_csv('{0}/{1}.csv'.format(write_path,metric))

def data_operate():
    results_path = '../../output_rank/'
    #metrics = ['Accuracy', 'Precision', 'Recall']
    metrics = ['Precision', 'Recall', 'F1', 'PofB', 'PMI', 'Popt', 'IFA']
    #metrics = ['Precision']
    for dataset in os.listdir(results_path):
        #if dataset == 'KNN':
        for metric in metrics:
            dataset_path = results_path + '%s/' % dataset
            print(dataset_path)
            datas = read_data(dataset_path, metric)
            combineData(datas, dataset, metric)

if __name__ == '__main__':
    data_operate()