import os
import time
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg
import sklearn.metrics
from Code.Processing import split_data2
from Code.classification import logistic_regression_classifier, knn_classifier, naive_bayes_classifier, \
    random_forest_classifier, decision_tree_classifier, svm_classifier, mlp_classifier


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K

class JDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lamJDA value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T

    def fit_predict(self,classifier, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        global clf, Xt_new
        list_acc = []
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        M = 0
        Y_tar_pseudo = None
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    tt = Ys == c
                    tmp = len(Ys[np.where(Ys == c)])
                    if tmp!=0:
                        e[np.where(tt == True)] = 1 / tmp
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    tmp=len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                    if tmp !=0 :
                        e[tuple(inds)] = -1 / tmp
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)
            M = M0 + N
            M = M / np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            clf = classifier(Xs_new, Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)
            #print('JDA iteration [{}/{}]: Acc: {:.4f}'.format(t + 1, self.T, acc))
        return clf,Xt_new


def JDA_func(classifier,train_data_x, train_data_y, test_data_x, test_data_y):
    print("JDA start time：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    starttime = time.time()
    Xs = train_data_x
    Ys = train_data_y
    Xt = test_data_x
    Yt = test_data_y

    jda = JDA(kernel_type='primal', dim=19, lamb=1, gamma=1)
    model = jda.fit_predict(classifier,Xs, Ys, Xt, Yt)

    return model
def save_Xt_new(data_count,classifier_name,Xt,Yt):
    Xt = Xt.tolist()
    for x in range(len(Xt)):
        Xt[x].append(Yt[x])
    df = pd.DataFrame(data = Xt)
    if not os.path.exists("../Xt/Data{0}".format(data_count)):
        os.makedirs("../Xt/Data{0}".format(data_count))
    df.to_csv('../Xt/Data{0}/JDA_Xt_{1}.csv'.format(data_count,classifier_name))

def JDA_func2(data_count,classify_name,Xs, Ys, Xt, Yt):
    jda = JDA(kernel_type='primal', dim=19, lamb=1, gamma=1)
    model, Xt_new = jda.fit_predict(classifiers[classify_name],Xs, Ys, Xt, Yt)
    save_Xt_new(data_count,classify_name,Xt_new,Yt)

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
            classifiers_name = ['LR', 'NB', 'KNN', 'RF', 'DT', 'MLP']
            data_count += 1
            if data_count>=8:
                for cl_name in classifiers_name:
                    JDA_func2(data_count,cl_name,train_data_x,train_data_y,test_data_x, test_data_y)