import pandas as pd
import time


def CBSplus(classifier,train_data_x, train_data_y,test_data_x, test_data_y):
    train_data_x = pd.DataFrame(train_data_x)
    train_data_y = pd.DataFrame(train_data_y)
    train_data_y = train_data_y.astype('int')
    classify_model = classifier(train_data_x, train_data_y)

    return classify_model


