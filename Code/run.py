import os
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
from Code.TCA import TCA_func
from Code.TNB import TNB
from Code.classification import naive_bayes_classifier, logistic_regression_classifier, random_forest_classifier, \
    decision_tree_classifier, knn_classifier, svm_classifier, mlp_classifier

warnings.filterwarnings("ignore")

def result_output(result_list, model_name,classifier):
    df = pd.DataFrame(data=result_list, columns=['Precision', 'Recall', 'F1', 'PofB', 'PMI', 'Popt', 'IFA'],index=None)
    if not os.path.exists("../output_rank/{0}".format(classifier)):
        os.makedirs("../output_rank/{0}".format(classifier))
    df.to_csv("../output_rank/{0}/{1}.csv".format(classifier,model_name),index=False)

def result_output_classify(result_list, model_name,classifier):
    df = pd.DataFrame(data=result_list, columns=['Precision', 'Recall', 'F1'],index=None)
    # 将结果保存到csv文件中

    if not os.path.exists("../output_classify/{0}".format(classifier)):
        os.makedirs("../output_classify/{0}".format(classifier))
    df.to_csv("../output_classify/{0}/{1}.csv".format(classifier, model_name), index=False)

def train_model(count,func,classifier,train_data_x, train_data_y, test_data_x, test_data_y):

    if func=='BDA':
        model = functions[func](count,classifier,train_data_x, train_data_y,test_data_x,test_data_y)
    else:
        model = functions[func](classifiers[classifier], train_data_x, train_data_y, test_data_x, test_data_y)
    path='../model/Data{0}/{1}'.format(count,func)
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

    joblib.dump(model, '{0}/{1}.pkl'.format(path,classifier))

def load_model(count,func,classifier,number):
    f_name = func[:-1]
    if f_name == 'None':
        f_name = 'CBSplus'
    if number in ['1','2','3','4']:
        path = '../model/Data{0}/{1}/{2}.pkl'.format(count,f_name, classifier)
    else:
        path = '../model/Data{0}/{1}/{2}.pkl'.format(count, func, classifier)
    model = joblib.load(path)
    return model

def split_Xt(dataset):
    dataset = np.array(dataset,dtype=float)
    k = len(dataset[0])
    y = dataset[:, k - 1]
    tmp_x = dataset[:, 0:k - 1]
    x = np.delete(tmp_x, 0, axis=1)
    return x, y

def assess_model_transfer(model,data_count,function,classify,loc,number):
    #This function is used to calculate evaluation metrics and evaluate the transfer learning models.
    file_path = '../Xt/Data{0}/{1}_Xt_{2}.csv'.format(data_count,function,classify)
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
        assessment = ranking_cbs(Yt, prob, loc)
    elif number == '4':
        assessment = ranking_prob(Yt, prob, loc)
    else:
        assessment = evaluate_classify(Yt, pred)
    return assessment

def assess_model(model,Xt,Yt,loc,number):
    # This function is used to calculate evaluation metrics and evaluate the data filtering models.
    pred = model.predict(Xt)
    probs = model.predict_proba(Xt)
    prob = [(1-p[0]) for p in probs]
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

def tarin_model(data_path):
    # Perform cross-project defect prediction Each of the 11 datasets is used as a test set once,
    # and the other datasets are used as training sets.
    # There are 11 training test set pairs
    data_count = 0
    for root, dirs, files, in os.walk(data_path):
        for file in files:
            # Get test set paths and data
            file_path = os.path.join(data_path,file)
            dataset_test = pd.read_csv(file_path)
            # Obtain training set paths and data
            dataset_train = pd.DataFrame(columns=dataset_test.columns)
            for tmp_file in files:
                if tmp_file == file:
                    continue
                else:
                    tmp_file_path = os.path.join(data_path,tmp_file)
                    tmp_df = pd.read_csv(tmp_file_path)
                    dataset_train = pd.concat([dataset_train, tmp_df])
            # Extract the x, y, loc of the dataset and return the data in ndarray format
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
    data_path = "../Data"
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
        'TCA':TCA_func,
        'BDA':BDA_func,
        'JDA':JDA_func,
        'JPDA':JPDA_func,
        'TNB':TNB
    }

    classifiers_name = ['LR', 'NB', 'KNN', 'RF', 'DT', 'MLP']
    # Note TNB algorithm to be run separately, because the method does not use any existing classifier,
    # but is the result of its own improvement.

    # Training model opens both, followed by all comments.
    #If you want to avoid repeated training of the model, be sure to comment the following two lines of code.
    functions_name = ['CBSplus', 'BF', 'PF', 'KF', 'DFAC', 'TCA', 'BDA', 'JDA', 'JPDA']
    tarin_model(data_path)

    # When calculating the indexes, if you use the sorting, one by one, the comments.
    # 1,2,3,4 represent each method of calculating the density of defects.
    # Note that only one functions_name can be uncommented at a time.
    # Ranking task:
    #functions_name = ['None1','BF1','PF1','KF1','DFAC1','TCA1','BDA1','JDA1','JPDA1']
    #functions_name = ['None2', 'BF2', 'PF2', 'KF2', 'DFAC2', 'TCA2', 'BDA2', 'JDA2', 'JPDA2']
    #functions_name = ['None3', 'BF3', 'PF3', 'KF3', 'DFAC3', 'TCA3', 'BDA3', 'JDA3', 'JPDA3']
    #functions_name = ['None4', 'BF4', 'PF4', 'KF4', 'DFAC4', 'TCA4', 'BDA4', 'JDA4', 'JPDA4']

    # classificaition task:
    #functions_name = ['CBSplus', 'BF', 'PF', 'KF', 'DFAC', 'TCA', 'BDA', 'JDA', 'JPDA']
    tarin_model(data_path)
    estimate_model(data_path)

