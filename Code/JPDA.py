import os
import time
import joblib
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.preprocessing import OneHotEncoder

from Code.Processing import split_data2
from Code.classification import logistic_regression_classifier, naive_bayes_classifier, knn_classifier, \
    random_forest_classifier, decision_tree_classifier, svm_classifier, mlp_classifier


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


def get_matrix_M(Ys, Y_tar_pseudo, ns, nt, C, mu, type='djp-mmd'):
    M = 0
    if type == 'jmmd':
        N = 0
        n = ns + nt
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M0 = e * e.T * C
        if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
            for c in range(1, C + 1):
                e = np.zeros((n, 1))
                tt = Ys == c
                e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])
                yy = Y_tar_pseudo == c
                ind = np.where(yy == True)
                inds = [item + ns for item in ind]
                e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                e[np.isinf(e)] = 0
                N = N + np.dot(e, e.T)
        M = M0 + N
        M = M / np.linalg.norm(M, 'fro')

    if type == 'jp-mmd':
        ohe = OneHotEncoder()
        ohe.fit(np.unique(Ys).reshape(-1, 1))
        Ys_ohe = ohe.transform(Ys.reshape(-1, 1)).toarray().astype(np.int8)

        # For transferability
        Ns = 1 / ns * Ys_ohe
        Nt = np.zeros([nt, C])
        if Y_tar_pseudo is not None:
            Yt_ohe = ohe.transform(Y_tar_pseudo.reshape(-1, 1)).toarray().astype(np.int8)
            Nt = 1 / nt * Yt_ohe
        Rmin = np.r_[np.c_[np.dot(Ns, Ns.T), np.dot(-Ns, Nt.T)], np.c_[np.dot(-Nt, Ns.T), np.dot(Nt, Nt.T)]]
        M = Rmin / np.linalg.norm(Rmin, 'fro')

    if type == 'djp-mmd':
        ohe = OneHotEncoder()
        ohe.fit(np.unique(Ys).reshape(-1, 1))
        Ys_ohe = ohe.transform(Ys.reshape(-1, 1)).toarray().astype(np.int8)

        # For transferability
        Ns = 1 / ns * Ys_ohe
        Nt = np.zeros([nt, C])
        if Y_tar_pseudo is not None:
            Yt_ohe = ohe.transform(Y_tar_pseudo.reshape(-1, 1)).toarray().astype(np.int8)
            Nt = 1 / nt * Yt_ohe
        Rmin = np.r_[np.c_[np.dot(Ns, Ns.T), np.dot(-Ns, Nt.T)], np.c_[np.dot(-Nt, Ns.T), np.dot(Nt, Nt.T)]]
        Rmin = Rmin / np.linalg.norm(Rmin, 'fro')

        # For discriminability
        Ms = np.zeros([ns, (C - 1) * C])
        Mt = np.zeros([nt, (C - 1) * C])
        for i in range(C):
            idx = np.arange((C - 1) * i, (C - 1) * (i + 1))
            Ms[:, idx] = np.tile(Ns[:, i], (C - 1, 1)).T
            tmp = np.arange(C)
            Mt[:, idx] = Nt[:, tmp[tmp != i]]
        Rmax = np.r_[np.c_[np.dot(Ms, Ms.T), np.dot(-Ms, Mt.T)], np.c_[np.dot(-Mt, Ms.T), np.dot(Mt, Mt.T)]]
        Rmax = Rmax / np.linalg.norm(Rmax, 'fro')
        M = Rmin - mu * Rmax

    return M


class DA_statistics:
    def __init__(self, kernel_type='primal', mmd_type='djp-mmd', dim=30, lamb=1, gamma=1, mu=0.1, T=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lamJPDA value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.mmd_type = mmd_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.mu = mu
        self.T = T

    def fit_predict(self,classifier, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JPDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        global clf, Xt_new
        X = np.hstack((Xs.T, Xt.T))
        X = np.dot(X, np.diag(1. / np.linalg.norm(X,axis=0)))
        m, n = X.shape  # 800, 2081
        ns, nt = len(Xs), len(Xt)

        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        Y_tar_pseudo = None
        list_acc = []
        for itr in range(self.T):
            M = get_matrix_M(Ys, Y_tar_pseudo, ns, nt, C, self.mu, type=self.mmd_type)

            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            clf = classifier(Xs_new,Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            acc = accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)
            print('iteration [{}/{}]: acc: {:.4f}'.format(itr + 1, self.T, acc))
        return clf,Xt_new


def JPDA_func(classifier,train_data_x, train_data_y, test_data_x, test_data_y):
    starttime = time.time()
    Xs = train_data_x
    Ys = train_data_y
    Xt = test_data_x
    Yt = test_data_y
    ker_type = 'linear'
    mmd_type='djp-mmd'
    jpda = DA_statistics(kernel_type=ker_type, mmd_type=mmd_type, dim=19, lamb=1, gamma=1)
    model = jpda.fit_predict(classifier,Xs, Ys, Xt, Yt)

    return model

def save_Xt_new(data_count,classifier_name,Xt,Yt):

    Xt = Xt.tolist()
    for x in range(len(Xt)):
        Xt[x].append(Yt[x])
    df = pd.DataFrame(data = Xt)
    if not os.path.exists("../Xt/Data{0}".format(data_count)):
        os.makedirs("../Xt/Data{0}".format(data_count))
    df.to_csv('../Xt/Data{0}/JPDA_Xt_{1}.csv'.format(data_count,classifier_name))

def JPDA_func2(data_count,classify_name,Xs, Ys, Xt, Yt):
    Xs = train_data_x
    Ys = train_data_y
    Xt = test_data_x
    Yt = test_data_y
    ker_type = 'linear'
    mmd_type = 'djp-mmd'
    jpda = DA_statistics(kernel_type=ker_type, mmd_type=mmd_type, dim=19, lamb=1, gamma=1)
    model ,Xt_new= jpda.fit_predict(classifiers[classify_name], Xs, Ys, Xt, Yt)
    save_Xt_new(data_count,classify_name,Xt_new,Yt)
    return model



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
            train_data_x, train_data_y, train_data_loc = split_data2(dataset_train)
            test_data_x, test_data_y, test_data_loc = split_data2(dataset_test)
            classifiers_name = ['LR']
            data_count += 1
            if data_count == 6:
                for cl_name in classifiers_name:
                    model = JPDA_func2(data_count, cl_name, train_data_x, train_data_y, test_data_x, test_data_y)
                    path = '../model/Data{0}/JPDA'.format(data_count)
                    isExists = os.path.exists(path)
                    if not isExists:
                        os.makedirs(path)
                    joblib.dump(model, '{0}../{1}.pkl'.format(path, cl_name))

