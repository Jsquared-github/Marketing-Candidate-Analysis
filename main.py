import numpy
import pandas as pd
import matplotlib as mpl
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder

raw = pd.read_csv('Data\marketing_campaign.csv', delimiter='\t')

# Functions to Processes Data


def one_hot_encode(original_df, feature):
    encoded = pd.get_dummies(original_df[feature])
    res = pd.concat([original_df, encoded], axis=1)
    return (res)


def trasform_education(df):
    df.loc[df.Education == 'Basic', 'Education'] = 'High School'
    df.loc[df.Education == '2n Cycle', 'Education'] = 'Master'
    df.loc[df.Education == 'Graduation', 'Education'] = 'Bachelors'
    df['Education'] = df['Education'].map({'High School': 0, 'Bachelors': 1, 'Master': 2, 'PhD': 3})


def trasform_marital_status(df):
    df.loc[df.Marital_Status == 'YOLO', 'Marital_Status'] = 'Single'
    df.loc[df.Marital_Status == 'Absurd', 'Marital_Status'] = 'Single'
    df.loc[df.Marital_Status == 'Alone', 'Marital_Status'] = 'Single'
    df.loc[df.Marital_Status == 'Together', 'Marital_Status'] = 'Dating'


def drop_redundants(df):
    df.drop('Dt_Customer', axis=1, inplace=True)
    df.drop('Z_CostContact', axis=1, inplace=True)
    df.drop('Z_Revenue', axis=1, inplace=True)
    df.drop('Marital_Status', axis=1, inplace=True)
    df.drop('ID', axis=1, inplace=True)


def process_data(df):
    df.Year_Birth = 2022 - df.Year_Birth
    trasform_education(df)
    trasform_marital_status(df)
    df = one_hot_encode(df, 'Marital_Status')
    drop_redundants(df)

    return df


df = process_data(raw)
print(df)
