from os.path import isfile
import pandas as pd
import numpy as np

label_name = 'income'
categorical_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
continuous_names = ['age', 'fnlwgt', 'education-num', 'capital-loss', 'capital-gain', 'hours-per-week']
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
         'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', label_name
         ]

train_fp = './assets/csv/train.csv'
test_fp = './assets/csv/test.csv'
raw_train_fp = './assets/csv/adult.data'
raw_test_fp = './assets/csv/adult.test'


binarized_data_names = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'income', 'workclass_ Federal-gov', 'workclass_ Local-gov', 'workclass_ Private', 'workclass_ Self-emp-inc', 'workclass_ Self-emp-not-inc', 'workclass_ State-gov', 'workclass_ Without-pay', 'education_ 10th', 'education_ 11th', 'education_ 12th', 'education_ 1st-4th', 'education_ 5th-6th', 'education_ 7th-8th', 'education_ 9th', 'education_ Assoc-acdm', 'education_ Assoc-voc', 'education_ Bachelors', 'education_ Doctorate', 'education_ HS-grad', 'education_ Masters', 'education_ Preschool', 'education_ Prof-school', 'education_ Some-college', 'marital-status_ Divorced', 'marital-status_ Married-AF-spouse', 'marital-status_ Married-civ-spouse', 'marital-status_ Married-spouse-absent', 'marital-status_ Never-married', 'marital-status_ Separated', 'marital-status_ Widowed', 'occupation_ Adm-clerical', 'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving', 'relationship_ Husband', 'relationship_ Not-in-family', 'relationship_ Other-relative',
                        'relationship_ Own-child', 'relationship_ Unmarried', 'relationship_ Wife', 'race_ Amer-Indian-Eskimo', 'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other', 'race_ White', 'sex_ Female', 'sex_ Male', 'native-country_ Cambodia', 'native-country_ Canada', 'native-country_ China', 'native-country_ Columbia', 'native-country_ Cuba', 'native-country_ Dominican-Republic', 'native-country_ Ecuador', 'native-country_ El-Salvador', 'native-country_ England', 'native-country_ France', 'native-country_ Germany', 'native-country_ Greece', 'native-country_ Guatemala', 'native-country_ Haiti', 'native-country_ Honduras', 'native-country_ Hong', 'native-country_ Hungary', 'native-country_ India', 'native-country_ Iran', 'native-country_ Ireland', 'native-country_ Italy', 'native-country_ Jamaica', 'native-country_ Japan', 'native-country_ Laos', 'native-country_ Mexico', 'native-country_ Nicaragua', 'native-country_ Outlying-US(Guam-USVI-etc)', 'native-country_ Peru', 'native-country_ Philippines', 'native-country_ Poland', 'native-country_ Portugal', 'native-country_ Puerto-Rico', 'native-country_ Scotland', 'native-country_ South', 'native-country_ Taiwan', 'native-country_ Thailand', 'native-country_ Trinadad&Tobago', 'native-country_ United-States', 'native-country_ Vietnam', 'native-country_ Yugoslavia', 'native-country_ Holand-Netherlands']

categorical_dummy_names = list(set(binarized_data_names) - set(continuous_names) - set([label_name]))


def get_data():
    if isfile(train_fp) and isfile(test_fp):
        train = pd.read_csv(train_fp, dtype=np.int8)
        test = pd.read_csv(test_fp, dtype=np.int8)
    else:
        train_df = pd.read_csv(raw_train_fp, names=names, low_memory=False).dropna()
        test_df = pd.read_csv(raw_test_fp, names=names, low_memory=False).dropna()

        train = pd.get_dummies(train_df, columns=categorical_names)
        test = pd.get_dummies(test_df, columns=categorical_names)

        missing = set(train) - set(test)
        if (len(missing) > 0):
            for attr in missing:
                test[attr] = 0

        for attr in continuous_names:
            all_attr_values = list(train_df[attr].values) + list(train_df[attr].values)
            attr_mean = sum(all_attr_values) / len(all_attr_values)
            train[attr] = np.where(train_df[attr] <= attr_mean, 0, 1)
            test[attr] = np.where(test_df[attr] <= attr_mean, 0, 1)

        print('mapping label values to 0, 1')
        train[label_name] = train_df[label_name].map({' <=50K': 0, ' >50K': 1})
        test[label_name] = train_df[label_name].map({' <=50K': 0, ' >50K': 1})

        print('writing parsed data to new files.')
        train.to_csv(train_fp, index=False)
        test.to_csv(test_fp, index=False)

    train_label = train.income
    test_label = test.income

    return train.drop(columns=label_name), train_label, test.drop(columns=label_name), test_label


train_samples, train_label, test_samples, test_label = get_data()
train_samples_cat = train_samples[categorical_dummy_names]
test_samples_cat = test_samples[categorical_dummy_names]
