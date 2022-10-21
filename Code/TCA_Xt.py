import os
import time
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg
import sklearn.metrics

from Code.Processing import split_data
from Code.classification import logistic_regression_classifier, naive_bayes_classifier, knn_classifier, \
    random_forest_classifier, decision_tree_classifier, svm_classifier, mlp_classifier


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=19, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lamTCA value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        #X /= np.linalg.norm(X, axis=0)
        X /= np.linalg.norm(X)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self,classifier, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = classifier(Xs_new, Ys.ravel())

        return clf,Xt_new

def save_Xt_new(data_count,classifier_name,Xt,Yt):

    Xt = Xt.tolist()
    for x in range(len(Xt)):
        Xt[x].append(Yt[x])
    df = pd.DataFrame(data = Xt)
    if not os.path.exists("./Xt/Data{0}".format(data_count)):
        os.makedirs("./Xt/Data{0}".format(data_count))
    df.to_csv('./Xt/Data{0}/TCA_Xt_{1}.csv'.format(data_count,classifier_name))

def TCA_func(data_count,classify_name,train_data_x, train_data_y, test_data_x, test_data_y):
    tca = TCA(kernel_type='linear', dim=19, lamb=1, gamma=1)
    model ,Xt_new = tca.fit_predict(classifiers[classify_name],train_data_x, train_data_y, test_data_x,test_data_y)
    save_Xt_new(data_count, classify_name, Xt_new, test_data_y)
    #return model
if __name__ == '__main__':
    classifiers = {
        'LR': logistic_regression_classifier,
        'NB': naive_bayes_classifier,
        'KNN': knn_classifier,
        'RF': random_forest_classifier,
        'DT': decision_tree_classifier,
        'SVM': svm_classifier,
        'MLP': mlp_classifier
    }
    data_path = "./Data"
    data_count=0
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
            train_data_x, train_data_y, train_data_loc = split_data(dataset_train)
            test_data_x, test_data_y, test_data_loc = split_data(dataset_test)
            classifiers_name = ['DT', 'MLP']
            data_count += 1
            if data_count == 4:
                for cl_name in classifiers_name:
                    TCA_func(data_count,cl_name,train_data_x,train_data_y,test_data_x, test_data_y)