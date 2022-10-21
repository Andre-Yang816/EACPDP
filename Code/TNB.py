import copy
import os
import time
import numpy as np
import pandas as pd

from Code.Evaluate import ranking_label_loc, ranking_prob_loc, ranking_cbs, ranking_prob
from Code.Evaluate_classify import evaluate_classify
from Code.Processing import split_data2


def TNB(train_data_x, train_data_y, test_data_x, test_data_y):
    max=copy.copy(test_data_x[0])
    min=copy.copy(test_data_x[0])
    for sample in test_data_x[1:]:
        for feature in range(len(sample)):
            if sample[feature] > max[feature]:
                max[feature] = sample[feature]
            elif sample[feature] < min[feature]:
                min[feature] = sample[feature]
    ns=len(train_data_x)
    m=len(train_data_x[0])
    s = [0]*ns
    for i in range(ns):
        for j in range(m):
            if train_data_x[i][j] >= min[j] and train_data_x[i][j] <= max[j]:
                s[i] = s[i]+1
    w = [0.0] * ns
    for i in range(ns):
        w[i]=float(s[i]/pow((m-s[i]+1),2))
    sum_w=sum(w)
    nc=2.0
    sum_wc=0.0
    for i in range(len(train_data_y)):
        if train_data_y[i] == 0:
            sum_wc = sum_wc + w[i]

    p_false = (sum_wc + 1)/(sum_w + nc)
    p_true = 1-p_false

    n_class=[0] * m
    for j in range(m):
        temp=[i[j] for i in train_data_x]
        n_class[j]=len(set(temp))

    test_p_false=[0.0]*len(test_data_x)
    for sample_test in range(len(test_data_x)):
        p_feature_false = [0.0] * m
        p_feature_true = [0.0] * m
        for feature in range(m):
            false_sum_wv = 0.0
            true_sum_wv = 0.0
            for sample_train in range(len(train_data_x)):
                a = train_data_x[sample_train][feature]
                b = test_data_x[sample_test][feature]
                if a == b:
                    if train_data_y[sample_train] == 0:
                        false_sum_wv = false_sum_wv + w[sample_train]
                    else:
                        true_sum_wv = true_sum_wv + w[sample_train]
            p_feature_false[feature] = (false_sum_wv + 1)/(sum_wc + n_class[feature])
            p_feature_true[feature] = (true_sum_wv + 1 )/(sum_wc + n_class[feature])

        test_p_false[sample_test] = p_false*np.prod(p_feature_false)/(p_false*np.prod(p_feature_false)+p_true*np.prod(p_feature_true))
    prob = [0.0]*len(test_data_x)
    for i in range(len(test_p_false)):
        prob[i] = 1 - test_p_false[i]
    pred = [1 if prob[i]>0.5 else 0 for i in range(len(prob))]
    endtime = time.time()
    print("****************************************************" + '\n')

    return prob,pred

def result_output(result_list, model_name,classifier='KNN'):
    df = pd.DataFrame(data=result_list, columns=['Precision', 'Recall', 'F1', 'PofB', 'PMI', 'Popt', 'IFA'])

    if not os.path.exists("../output_rank/{0}".format(classifier)):
        os.makedirs("../output_rank/{0}".format(classifier))
    df.to_csv("../output_rank/{0}/{1}.csv".format(classifier,model_name),index=False)

def result_output_classify(result_list, model_name='TNB',classifier='KNN'):
    df = pd.DataFrame(data=result_list, columns=['Precision', 'Recall', 'F1'])
    if not os.path.exists("../output_classify/{0}".format(classifier)):
        os.makedirs("../output_classify/{0}".format(classifier))
    df.to_csv("../output_classify/{0}/{1}.csv".format(classifier, model_name), index=False)


if __name__ == '__main__':
    data_path = '../Data'
    # number：0 classification，1 label/loc,2 prob/loc，3 CBS+，4 prob
    #numbers = [1,2,3,4]
    numbers = [0]
    for number in numbers:
        result = []
        for root, dirs, files, in os.walk(data_path):
            for file in files:
                file_path = os.path.join(data_path, file)
                dataset_test = pd.read_csv(file_path)
                dataset_train = pd.DataFrame(columns=dataset_test.columns)
                for tmp_file in files:
                    if tmp_file == file:
                        continue
                    else:
                        tmp_file_path = os.path.join(data_path, tmp_file)
                        tmp_df = pd.read_csv(tmp_file_path)
                        dataset_train = pd.concat([dataset_train, tmp_df])
                Xs, Ys, train_data_loc = split_data2(dataset_train)
                Xt, Yt, LOC = split_data2(dataset_test)

                prob, pred = TNB(Xs, Ys, Xt, Yt)
                if number == 1:
                    assessment = ranking_label_loc(Yt, pred, LOC)
                elif number == 2:
                    assessment = ranking_prob_loc(Yt, prob, LOC)
                elif number == 3:
                    assessment = ranking_cbs(Yt, prob, LOC)
                elif number == 4:
                    assessment = ranking_prob(Yt, prob, LOC)
                else:
                    assessment = evaluate_classify(Yt, pred)
                result.append(assessment)
        if number == 0:
            result_output_classify(result)
        else:
            name = 'TNB{0}'.format(number)
            result_output(result, name)