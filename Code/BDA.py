import os
import time
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from sklearn import metrics
from sklearn import svm

from Code.Processing import split_data
from Code.classification import logistic_regression_classifier, knn_classifier, random_forest_classifier, \
    naive_bayes_classifier, decision_tree_classifier, mlp_classifier


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


def proxy_a_distance(source_X, target_X):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    train_X = np.vstack((source_X, target_X))
    train_Y = np.hstack((np.zeros(nb_source, dtype=int),
                         np.ones(nb_target, dtype=int)))

    clf = svm.LinearSVC()
    clf.fit(train_X, train_Y)
    y_pred = clf.predict(train_X)
    error = metrics.mean_absolute_error(train_Y, y_pred)
    dist = 2 * (1 - 2 * error)
    return dist


def estimate_mu(_X1, _Y1, _X2, _Y2):
    adist_m = proxy_a_distance(_X1, _X2)
    C = len(np.unique(_Y1))
    epsilon = 1e-3
    list_adist_c = []
    for i in range(1, C + 1):
        ind_i, ind_j = np.where(_Y1 == i), np.where(_Y2 == i)
        Xsi = _X1[ind_i[0], :]
        Xtj = _X2[ind_j[0], :]
        adist_i = proxy_a_distance(Xsi, Xtj)
        list_adist_c.append(adist_i)
    adist_c = sum(list_adist_c) / C
    mu = adist_c / (adist_c + adist_m)
    if mu > 1:
        mu = 1
    if mu < epsilon:
        mu = 0
    return mu


class BDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=0.5, mu=0.5, gamma=1, T=5, mode='BDA', estimate_mu=False):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param mu: mu. Default is -1, if not specificied, it calculates using A-distance
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        :param mode: 'BDA' | 'WBDA'
        :param estimate_mu: True | False, if you want to automatically estimate mu instead of manally set it
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.mu = mu
        self.gamma = gamma
        self.T = T
        self.mode = mode
        self.estimate_mu = estimate_mu

    def fit_predict(self, classifier,Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        global clf, Xt_new, Y_tar_prob
        list_acc = []
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))
        mu = self.mu
        M = 0
        Y_tar_pseudo = None
        Xs_new = None
        pred = None
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    Ns = len(Ys[np.where(Ys == c)])
                    Nt = len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])

                    if self.mode == 'WBDA':
                        Ps = Ns / len(Ys)
                        Pt = Nt / len(Y_tar_pseudo)
                        alpha = Pt / Ps
                        mu = 1
                    else:
                        alpha = 1

                    tt = Ys == c
                    if Ns!=0:
                        e[np.where(tt == True)] = 1 / Ns
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    if Nt!=0:
                        e[tuple(inds)] = -alpha / Nt
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)

            # In BDA, mu can be set or automatically estimated using A-distance
            # In WBDA, we find that setting mu=1 is enough
            if self.estimate_mu and self.mode == 'BDA':
                if Xs_new is not None:
                    mu = estimate_mu(Xs_new, Ys, Xt_new, Y_tar_pseudo)
                else:
                    mu = 0
            M = (1 - mu) * M0 + mu * N
            M /= np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T

            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = A.T @ K
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            clf = classifier(Xs_new, Ys.ravel())
            Y_tar_prob = clf.predict_proba(Xt_new)
            Y_tar_pseudo = clf.predict(Xt_new)
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)
            print('{} iteration [{}/{}]: Acc: {:.4f}'.format(self.mode, t + 1, self.T, acc))

        return clf,Xt_new

def save_Xt_new(data_count,classifier_name,Xt,Yt):
    Xt = Xt.tolist()
    for x in range(len(Xt)):
        Xt[x].append(Yt[x])
    df = pd.DataFrame(data = Xt)
    # 将结果保存到csv文件中
    if not os.path.exists("../Xt/Data{0}".format(data_count)):
        os.makedirs("../Xt/Data{0}".format(data_count))
    df.to_csv('../Xt/Data{0}/BDA_Xt_{1}.csv'.format(data_count,classifier_name))
def BDA_func2(data_count,classifier,Xs, Ys, Xt, Yt):
    print("BDA start time：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    starttime = time.time()

    bda = BDA(kernel_type='primal', dim=19, lamb=0.5, mu=0.5,
              mode='BDA', gamma=1, estimate_mu=False)

    model, Xt_new = bda.fit_predict(classifier,Xs, Ys, Xt, Yt)
    return model
def BDA_func(data_count,classify_name,Xs, Ys, Xt, Yt):
    print("BDA start time：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    starttime = time.time()
    classifiers = {
        'LR': logistic_regression_classifier,
        'NB': naive_bayes_classifier,
        'KNN': knn_classifier,
        'RF': random_forest_classifier,
        'DT': decision_tree_classifier,
        'MLP': mlp_classifier
    }
    bda = BDA(kernel_type='primal', dim=19, lamb=0.5, mu=0.5,
              mode='BDA', gamma=1, estimate_mu=False)

    model, Xt_new = bda.fit_predict(classifiers[classify_name],Xs, Ys, Xt, Yt)
    save_Xt_new(data_count,classify_name,Xt_new,Yt)
    return model

if __name__ == '__main__':
    classifiers = {
        'LR': logistic_regression_classifier,
        'NB': naive_bayes_classifier,
        'KNN': knn_classifier,
        'RF': random_forest_classifier,
        'DT': decision_tree_classifier,
        'MLP': mlp_classifier
    }
    data_path = "../Data"
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
            classifiers_name = ['KNN']
            data_count += 1
            for cl_name in classifiers_name:
                BDA_func2(data_count,cl_name,train_data_x,train_data_y,test_data_x, test_data_y)