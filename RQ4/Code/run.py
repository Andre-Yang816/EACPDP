import os
from pathlib import Path

import joblib
import warnings

from Code.BDA import BDA_func
from Code.Evaluate import ranking_cbs, ranking_prob, ranking_label_loc, ranking_prob_loc
from Code.Evaluate_classify import evaluate_classify
from Code.JDA import JDA_func
from Code.JPDA import JPDA_func
from Code.Processing import *
from Code.CBSplus import CBSplus
from Code.BF_filter import BF_filter
from Code.PF_filter import PF_filter
from Code.KF_filter import KF_filter
from Code.DFAC_filter import DFAC_filter
from Code.CamargoCruz09 import CamargoCruz09
from Code.TCA import TCA_func
from Code.TNB import TNB
from Code.classification import naive_bayes_classifier, logistic_regression_classifier, random_forest_classifier, \
    decision_tree_classifier, knn_classifier, mlp_classifier

warnings.filterwarnings("ignore")

def result_output(result_list, model_name,classifier,lamda):

    df = pd.DataFrame(data=result_list, columns=['Precision', 'Recall', 'F1', 'PofB', 'PMI', 'Popt', 'IFA'],index=None)
    median = []
    for i in range(df.shape[1]):
        median.append(df.iloc[:, i].median())
    Path("../output/median/{0}/l{1}/".format(classifier,lamda)).mkdir(parents=True, exist_ok=True)
    output_path = "../output/median/{0}/l{1}/{2}.csv".format(classifier,lamda, model_name)

    output = pd.DataFrame(data=[median], columns=['Precision', 'Recall', 'F1', 'PofB', 'PMI', 'Popt', 'IFA'],index=None)
    # output = pd.DataFrame(data=[pvalues], columns=functions)
    output.to_csv(output_path, encoding='utf-8')


def result_output_classify(result_list, model_name,classifier,lamda):
    df = pd.DataFrame(data=result_list, columns=['Precision', 'Recall', 'F1_measure'],index=None)
    median = []
    for i in range(df.shape[1]):
        median.append(df.iloc[:, i].median())

    Path("../output/classify/{0}/l{1}/".format(classifier, lamda)).mkdir(parents=True, exist_ok=True)
    output_path = "../cpdp_7classifier/RQ3/output/classify/{0}/l{1}/{2}.csv".format(classifier, lamda, model_name)

    output = pd.DataFrame(data=[median], columns=['Precision', 'Recall', 'F1_measure'],index=None)
    # output = pd.DataFrame(data=[pvalues], columns=functions)
    output.to_csv(output_path, encoding='utf-8')

def load_model(count,func,classifier,number):
    f_name = func[:-1]
    if f_name == 'None':
        f_name = 'CBSplus'
    if number in ['1','2','3','4']:
        path = '../../model/Data{0}/{1}/{2}.pkl'.format(count,f_name, classifier)
    else:
        path = '../../model/Data{0}/{1}/{2}.pkl'.format(count, func, classifier)
    model = joblib.load(path)
    return model

def split_Xt(dataset):
    dataset = np.array(dataset,dtype=float)
    k = len(dataset[0])
    y = dataset[:, k - 1]
    tmp_x = dataset[:, 0:k - 1]
    x = np.delete(tmp_x, 0, axis=1)
    return x, y

def assess_model_transfer(model,data_count,function,classify,loc,number,lamda):
    # Compute evaluation metrics and evaluate models.
    # Migration learning methods require additional reading of post-migration data.

    file_path = '../../Xt/Data{0}/{1}_Xt_{2}.csv'.format(data_count,function,classify)
    dataset_test = pd.read_csv(file_path)
    Xt_new ,Yt = split_Xt(dataset_test)
    pred = model.predict(Xt_new)
    probs = model.predict_proba(Xt_new)
    prob = [p[1] for p in probs]
    if number == '1':
        assessment = ranking_label_loc(Yt, pred, loc)
    elif number == '2':
        assessment = ranking_prob_loc(Yt, prob, loc)
    elif number == '3':
        assessment = ranking_cbs(Yt, prob, loc, lamda)
    elif number == '4':
        assessment = ranking_prob(Yt, prob, loc)
    else:
        assessment = evaluate_classify(Yt, pred)
    return assessment

def assess_model(model,Xt,Yt,loc,number,lamd):
    assessment = []
    pred = model.predict(Xt)
    probs = model.predict_proba(Xt)
    prob = [(1-p[0]) for p in probs]
    #if number == 1,2,3,4:ranking merics，else classification metrics
    if number == '1':
        assessment = ranking_label_loc(Yt, pred, loc)
    elif number == '2':
        assessment = ranking_prob_loc(Yt, prob, loc)
    elif number == '3':
        assessment = ranking_cbs(Yt, prob, loc, lamd)
    elif number == '4':
        assessment = ranking_prob(Yt, prob, loc)
    else:
        assessment = evaluate_classify(Yt, pred)
    return assessment

def estimate_model(data_path,lamda):
    result=[]
    data_count = 0
    number = 0
    for root, dirs, files, in os.walk(data_path):
        for file in files:
            file_path = os.path.join(data_path,file)
            dataset_test = pd.read_csv(file_path)
            dataset_train = pd.DataFrame(columns=dataset_test.columns)
            for tmp_file in files:
                if tmp_file == file:
                    continue
                else:
                    tmp_file_path = os.path.join(data_path,tmp_file)
                    tmp_df = pd.read_csv(tmp_file_path)
                    dataset_train = pd.concat([dataset_train, tmp_df])

            train_data_x, train_data_y, train_data_loc = split_data2(dataset_train)
            test_data_x, test_data_y, test_data_loc = split_data2(dataset_test)

            data_count += 1
            tmp_f=[]
            for i in range(len(functions_name)):
                tmp_c=[]
                number = functions_name[i][-1]
                for j in range(len(classifiers_name)):
                    model = load_model(data_count, functions_name[i], classifiers_name[j],number)
                    if functions_name[i] in ['TCA','BDA','JDA','JPDA']:
                        assessment = assess_model_transfer(model,data_count,functions_name[i],classifiers_name[j],test_data_loc,number,lamda)
                    else:
                        assessment = assess_model(model,test_data_x,test_data_y,test_data_loc,number,lamda)
                    tmp_c.append(assessment)
                tmp_f.append(tmp_c)
            result.append(tmp_f)
    result = [[row[i] for row in result] for i in range(len(result[0]))]
    for func in range(len(functions_name)):
        tmp = [[row[i] for row in result[func]] for i in range(len(result[func][0]))]
        for cl in range(len(tmp)):
            name = functions_name[func]
            if number in ['1','2','3','4']:
                result_output(tmp[cl],name,classifiers_name[cl],lamda)
            else:
                result_output_classify(tmp[cl],name,classifiers_name[cl],lamda)

if __name__ == "__main__":
    data_path = "../../Data"
    classifiers = {
        'LR': logistic_regression_classifier,
        'NB': naive_bayes_classifier,
        'KNN':knn_classifier,
        'RF': random_forest_classifier,
        'DT': decision_tree_classifier,
        'MLP':mlp_classifier
    }
    functions = {
        'CBSplus':CBSplus,
        'BF':BF_filter,
        'PF':PF_filter,
        'KF':KF_filter,
        'DFAC':DFAC_filter,
        'CC':CamargoCruz09,
        'TCA':TCA_func,
        'BDA':BDA_func,
        'JDA':JDA_func,
        'JPDA':JPDA_func,
        'TNB':TNB
    }

    classifiers_name = ['KNN', 'LR', 'RF']
    functions_name = ['JDA3','BDA3']
    lamda = 0.5
    estimate_model(data_path,lamda)

