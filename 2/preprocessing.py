import numpy as np
import pandas as pd


def do_preprocessing(train_file: str, test_file: str, skip_initial_space=False, save_file=False):
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
        'relationship', 'race', 'sex', 'native-country', 'income'
    ]

    train = pd.read_csv(train_file, names=header, skipinitialspace=skip_initial_space, comment='|')

    test = pd.read_csv(test_file, names=header, skipinitialspace=skip_initial_space, comment='|')

    train = pd.get_dummies(train, columns=categorical_attributes)
    test = pd.get_dummies(test, columns=categorical_attributes)

    train[numeric_attributes] = np.where(train[numeric_attributes].mean() >= train[numeric_attributes], 0, 1)
    
    test[numeric_attributes] = np.where(test[numeric_attributes].mean() >= test[numeric_attributes], 0, 1)

    if save_file:
        train.to_csv('data/train')
        test.to_csv('data/test')
    else:
        return train, test

do_preprocessing('data/adult.data', 'data/adult.test', skip_initial_space=True, save_file=True)
