import pandas as pd
from datetime import datetime, date

raw = pd.read_csv('C:/Users/2021j/Documents/Data/marketing_campaign.csv', delimiter='\t')

# Functions to Processes Data


def one_hot_encode(original_df, feature):
    encoded = pd.get_dummies(original_df[feature])
    res = pd.concat([original_df, encoded], axis=1)
    return (res)


def condense_education(df):
    df.Education.loc[df['Education'] == 'Basic'] = 'High School'
    df.Education.loc[df['Education'] == '2n Cycle'] = 'Master'
    df.Education.loc[df['Education'] == 'Graduation'] = 'Bachelors'


def condesnse_marital_status(df):
    df.Marital_Status.loc[df['Marital_Status'] == 'YOLO'] = 'Single'
    df.Marital_Status.loc[df['Marital_Status'] == 'Absurd'] = 'Single'
    df.Marital_Status.loc[df['Marital_Status'] == 'Alone'] = 'Single'
    df.Marital_Status.loc[df['Marital_Status'] == 'Together'] = 'Dating'


def drop_redundants(df):
    df.drop('Dt_Customer', axis=1, inplace=True)
    df.drop('Z_CostContact', axis=1, inplace=True)
    df.drop('Z_Revenue', axis=1, inplace=True)
    df.drop('Education', axis=1, inplace=True)
    df.drop('Marital_Status', axis=1, inplace=True)


def process_data(df):
    df.Year_Birth = 2022 - df.Year_Birth
    condense_education(df)
    condesnse_marital_status(df)
    df = one_hot_encode(df, 'Education')
    df = one_hot_encode(df, 'Marital_Status')
    drop_redundants(df)

    print(df.info())


process_data(raw)
