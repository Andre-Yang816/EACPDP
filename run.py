import os
from pathlib import Path

import joblib
import warnings

from BDA import BDA_func
from Evaluate import ranking_cbs, ranking_prob, ranking_label_loc, ranking_prob_loc
from Evaluate_classify import evaluate_classify
from JDA import JDA_func
from JPDA import JPDA_func
from Processing import *
from CBSplus import CBSplus
from BF_filter import BF_filter
from PF_filter import PF_filter
from KF_filter import KF_filter
from DFAC_filter import DFAC_filter
from CamargoCruz09 import CamargoCruz09
from TCA import TCA_func
from Turhan import Turhan
from TNB import TNB
from Yu_filter import Yu_filter
from classification import naive_bayes_classifier, logistic_regression_classifier, random_forest_classifier, \
    decision_tree_classifier, knn_classifier, svm_classifier, mlp_classifier

warnings.filterwarnings("ignore")

def result_output(result_list, model_name,classifier):
    df = pd.DataFrame(data=result_list, columns=['Precision', 'Recall', 'F1', 'PofB', 'PMI', 'Popt', 'IFA'],index=None)
    if not os.path.exists("./output_rank/{0}".format(classifier)):
        os.makedirs("./output_rank/{0}".format(classifier))
    df.to_csv("./output_rank/{0}/{1}.csv".format(classifier,model_name),index=False)


def result_output_classify(result_list, model_name,classifier):
    df = pd.DataFrame(data=result_list, columns=['Precision', 'Recall', 'F1_measure'],index=None)
    # 将结果保存到csv文件中

    if not os.path.exists("./output_classify/{0}".format(classifier)):
        os.makedirs("./output_classify/{0}".format(classifier))
    df.to_csv("./output_classify/{0}/{1}.csv".format(classifier, model_name), index=False)


def train_model(count,func,classifier,train_data_x, train_data_y, test_data_x, test_data_y):
    print('{0} 模型 使用 {1} 分类器训练 在第{2}个数据集上：'.format(func,classifier,count))
    if func=='BDA':
        model = functions[func](count,classifier,train_data_x, train_data_y,test_data_x,test_data_y)
    else:
        model = functions[func](classifiers[classifier], train_data_x, train_data_y, test_data_x, test_data_y)
    path='./model/Data{0}/{1}'.format(count,func)
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

    joblib.dump(model, '{0}/{1}.pkl'.format(path,classifier))

def load_model(count,func,classifier,number):
    f_name = func[:-1]
    if f_name == 'None':
        f_name = 'CBSplus'
    if number in ['1','2','3','4']:
        path = './model/Data{0}/{1}/{2}.pkl'.format(count,f_name, classifier)
    else:
        path = './model/Data{0}/{1}/{2}.pkl'.format(count, func, classifier)
    model = joblib.load(path)
    return model

def split_Xt(dataset):    # 分离数据集的x，y和loc
    dataset = np.array(dataset,dtype=float)
    k = len(dataset[0])     # 数据列数
    y = dataset[:, k - 1]   # 数据缺陷真值
    tmp_x = dataset[:, 0:k - 1]     # 数据特征
    x = np.delete(tmp_x, 0, axis=1)
    return x, y

def assess_model_transfer(model,data_count,function,classify,loc,number):
    #计算评价指标，评估模型
    assessment = []
    print('数据集：Data{0}/{1}_{2}.csv'.format(data_count, function, classify))
    file_path = './Xt/Data{0}/{1}_Xt_{2}.csv'.format(data_count,function,classify)
    dataset_test = pd.read_csv(file_path)
    Xt_new ,Yt = split_Xt(dataset_test)
    pred = model.predict(Xt_new)
    probs = model.predict_proba(Xt_new)
    prob = [p[1] for p in probs]
    #计算评价指标
    if number == '1':
        assessment = ranking_label_loc(Yt, pred, loc)
    elif number == '2':
        assessment = ranking_prob_loc(Yt, prob, loc)
    elif number == '3':
        assessment = ranking_cbs(Yt, prob, loc)
    elif number == '4':
        assessment = ranking_prob(Yt, prob, loc)
    else:
        assessment = evaluate_classify(Yt, pred)
    return assessment

def assess_model(model,Xt,Yt,loc,number):
    #计算评价指标，评估模型
    assessment = []
    pred = model.predict(Xt)
    probs = model.predict_proba(Xt)
    prob = [(1-p[0]) for p in probs]
    #如果为1,2,3,4，则为排序指标，否则为分类指标
    if number == '1':
        assessment = ranking_label_loc(Yt, pred, loc)
    elif number == '2':
        assessment = ranking_prob_loc(Yt, prob, loc)
    elif number == '3':
        assessment = ranking_cbs(Yt, prob, loc)
    elif number == '4':
        assessment = ranking_prob(Yt, prob, loc)
    else:
        # 计算评价指标 ACC、Precision、Recall、F-measure、AUC
        assessment = evaluate_classify(Yt, pred)
    return assessment

def tarin_model(data_path):
    data_count = 0
    for root, dirs, files, in os.walk(data_path):
        # 执行跨项目缺陷预测 11个数据集各作为一次测试集，其他数据集作为训练集 共有11组训练测试集对
        for file in files:
            # 获取测试集路径和数据
            file_path = os.path.join(data_path,file)
            dataset_test = pd.read_csv(file_path)

            # 获取训练集路径和数据
            dataset_train = pd.DataFrame(columns=dataset_test.columns)
            for tmp_file in files:
                if tmp_file == file:    # 如果tmp_file是当前测试集，则跳过
                    continue
                else:
                    tmp_file_path = os.path.join(data_path,tmp_file)
                    tmp_df = pd.read_csv(tmp_file_path)
                    dataset_train = pd.concat([dataset_train, tmp_df])

            # 提取数据集的x, y, loc 返回的数据格式是ndarray
            train_data_x, train_data_y, train_data_loc = split_data(dataset_train)
            test_data_x, test_data_y, test_data_loc = split_data(dataset_test)


            data_count += 1

            for func in functions_name:
                for classifier in classifiers_name:
                    train_model(data_count,func,classifier,train_data_x, train_data_y, test_data_x, test_data_y)

def estimate_model(data_path):
    result=[]
    data_count = 0
    number = 0
    for root, dirs, files, in os.walk(data_path):
        # 执行跨项目缺陷预测 11个数据集各作为一次测试集，其他数据集作为训练集 共有11组训练测试集对
        for file in files:
            # 获取测试集路径和数据
            file_path = os.path.join(data_path,file)
            dataset_test = pd.read_csv(file_path)

            # 获取训练集路径和数据
            dataset_train = pd.DataFrame(columns=dataset_test.columns)
            for tmp_file in files:
                if tmp_file == file:    # 如果tmp_file是当前测试集，则跳过
                    continue
                else:
                    tmp_file_path = os.path.join(data_path,tmp_file)
                    tmp_df = pd.read_csv(tmp_file_path)
                    dataset_train = pd.concat([dataset_train, tmp_df])

            # 提取数据集的x, y, loc 返回的数据格式是ndarray,y为真实缺陷个数
            train_data_x, train_data_y, train_data_loc = split_data2(dataset_train)
            test_data_x, test_data_y, test_data_loc = split_data2(dataset_test)


            data_count += 1
            #number控制是排序还是分类
            tmp_f=[]
            for i in range(len(functions_name)):
                tmp_c=[]
                number = functions_name[i][-1]
                for j in range(len(classifiers_name)):

                    model = load_model(data_count, functions_name[i], classifiers_name[j],number)
                    print('方法：{0}，分类器：{1}，数据集：{2}'.format(functions_name[i], classifiers_name[j], data_count))

                    if functions_name[i] in ['TCA','BDA','JDA','JPDA']:
                        assessment = assess_model_transfer(model,data_count,functions_name[i],classifiers_name[j],test_data_loc,number)
                    else:
                        assessment = assess_model(model,test_data_x,test_data_y,test_data_loc,number)
                    tmp_c.append(assessment)
                tmp_f.append(tmp_c)
            result.append(tmp_f)
    result = [[row[i] for row in result] for i in range(len(result[0]))]
    for func in range(len(functions_name)):
        tmp = [[row[i] for row in result[func]] for i in range(len(result[func][0]))]
        for cl in range(len(tmp)):
            name = functions_name[func]
            if number in ['1','2','3','4']:
                result_output(tmp[cl],name,classifiers_name[cl])
            else:
                result_output_classify(tmp[cl],name,classifiers_name[cl])

if __name__ == "__main__":
    data_path = "./Data"
    classifiers = {
        'LR': logistic_regression_classifier,
        'NB': naive_bayes_classifier,
        'KNN':knn_classifier,
        'RF': random_forest_classifier,
        'DT': decision_tree_classifier,
        'SVM':svm_classifier,
        'MLP':mlp_classifier
    }
    functions = {
        'CBSplus':CBSplus,
        'BF':BF_filter,
        'PF':PF_filter,
        'KF':KF_filter,
        'DFAC':DFAC_filter,
        'CC':CamargoCruz09,
        'Turhan':Turhan,
        'Yu':Yu_filter,
        'TCA':TCA_func,
        'BDA':BDA_func,
        'JDA':JDA_func,
        'JPDA':JPDA_func,
        'TNB':TNB
    }

    classifiers_name = ['LR', 'NB', 'KNN', 'RF', 'DT', 'MLP']
    #classifiers_name = ['KNN']
    #注TNB算法要单独跑，因为该方法不使用任何现有分类器，而是自己改进的结果。
    #训练模型打开这两个，后面全部注释
    # functions_name = ['CBSplus', 'BF', 'PF', 'KF', 'DFAC', 'TCA', 'BDA', 'JDA', 'JPDA']
    # tarin_model(data_path)

    #classifiers_name = ['NB']
    # 计算指标的时候，使用排序的话，一个个解除注释,1,2,3,4代表各个计算缺陷密度的方法
    #functions_name = ['None1','BF1','PF1','KF1','DFAC1','TCA1','BDA1','JDA1','JPDA1']
    #functions_name = ['None2', 'BF2', 'PF2', 'KF2', 'DFAC2', 'TCA2', 'BDA2', 'JDA2', 'JPDA2']
    #functions_name = ['None3', 'BF3', 'PF3', 'KF3', 'DFAC3', 'TCA3', 'BDA3', 'JDA3', 'JPDA3']
    #functions_name = ['None4', 'BF4', 'PF4', 'KF4', 'DFAC4', 'TCA4', 'BDA4', 'JDA4', 'JPDA4']
    #分类任务，注意，一次只可以解除一个functions_name的注释
    #functions_name = ['CBSplus', 'BF', 'PF', 'KF', 'DFAC', 'TCA', 'BDA', 'JDA', 'JPDA']
    functions_name = ['BDA4']
    #tarin_model(data_path)
    estimate_model(data_path)

