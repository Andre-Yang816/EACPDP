from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


import warnings
warnings.filterwarnings("ignore")

def logistic_regression_classifier(train_X, train_y):
    """
    Logistic regression model
    :param train_X: the features of the training set
    :param train_y: the labels of the training set
    :return: the logistic regression model
    """
    model = LogisticRegression()
    model.fit(train_X, train_y)

    return model

def naive_bayes_classifier(train_X, train_y):
    """
    Plain Bayesian model
    :param train_X: the features of the training set
    :param train_y: the labels of the training set
    :return: plain Bayesian model
    """
    model = GaussianNB()
    model.fit(train_X, train_y)
    return model

def knn_classifier(train_X, train_y):
    """
    KNN
    :param train_X: the features of the training set
    :param train_y: the labels of the training set
    :return: KNN model
    """
    model = KNeighborsClassifier()
    model.fit(train_X, train_y)

    return model

def random_forest_classifier(train_X, train_y):
    """
    Random forest model
    :param train_X: the features of the training set
    :param train_y: the labels of the training set
    :return: the random forest model
    """
    model = RandomForestClassifier()
    model.fit(train_X, train_y)

    return model

def decision_tree_classifier(train_X, train_y):
    """
    Decision tree model
    :param train_X: the features of the training set
    :param train_y: the labels of the training set
    :return: the decision tree model
    """
    model = tree.DecisionTreeClassifier()
    model.fit(train_X, train_y)

    return model

def svm_classifier(train_X, train_y):
    """
     Support vector machine model
     :param train_X: the features of the training set
     :param train_y: the labels of the training set
     :return: SVM model
     """
    model = SVC()
    model.fit(train_X, train_y)

    return model

def mlp_classifier(train_X, train_y):
    """
    Multilayer Perceptron model
    :param train_X: the features of the training set
    :param train_y: the labels of the training set
    :return: plain MLP model
    """
    model = MLPClassifier()
    model.fit(train_X, train_y)

    return model