import numpy as np
import pandas as pd


def do_preprocessing(data_file: str, test_file: str, skip_initial_space=False,
                     save_file=False):
    header = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'income'
    ]
    
    numeric_attributes = [
        'age', 'fnlwgt', 'education-num', 'capital-loss', 'capital-gain',
        'hours-per-week'
    ]

    categorical_attributes = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]

    X = pd.read_csv(data_file, names=header,
                    skipinitialspace=skip_initial_space, comment='|')

    Y = pd.read_csv(test_file, names=header,
                    skipinitialspace=skip_initial_space, comment='|')

    X = pd.get_dummies(X, columns=categorical_attributes)
    Y = pd.get_dummies(Y, columns=categorical_attributes)

    X[numeric_attributes] = np.where(X[numeric_attributes].mean() >=
                                     X[numeric_attributes], 0, 1)
    
    Y[numeric_attributes] = np.where(Y[numeric_attributes].mean() >=
                                     Y[numeric_attributes], 0, 1)

    if save_file:
        X.to_csv('p_' + data_file)
        Y.to_csv('p_' + test_file)
    else:
        return X, Y

do_preprocessing('adult.data', 'adult.test', skip_initial_space=True, save_file=True)
