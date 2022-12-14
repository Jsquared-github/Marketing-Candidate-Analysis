import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, date
from sklearn.impute import SimpleImputer

raw = pd.read_csv('Data\marketing_campaign.csv', delimiter='\t')

# Functions to Processes Data


def impute_missing_data(df):
    mean_imputer = SimpleImputer(missing_values=pd.NA, strategy='mean')
    df.Income = mean_imputer.fit_transform(df[['Income']])


def remove_outliers(df, feature):
    Q1 = df[feature].describe()[6]
    Q3 = df[feature].describe()[4]
    IQR = Q1 - Q3
    df.drop(df[(df[feature] < (Q1 - 1.75 * IQR))].index, inplace=True)
    df.drop(df[(df[feature] > (Q3 + 1.75 * IQR))].index, inplace=True)


def transform_age(df):
    df['Age'] = 2022 - df.Year_Birth


def trasform_education(df):
    df.loc[df.Education == 'Basic', 'Education'] = 'High School'
    df.loc[df.Education == '2n Cycle', 'Education'] = 'Master'
    df.loc[df.Education == 'Graduation', 'Education'] = 'Bachelors'
    df['Education'] = df['Education'].map({'High School': 0, 'Bachelors': 1, 'Master': 2, 'PhD': 3})


def combine_campaigns(df):
    df['Campaigns_Accepted'] = df.AcceptedCmp1 + df.AcceptedCmp2 + df.AcceptedCmp3 + df.AcceptedCmp4 + df.AcceptedCmp5 + df.Response


def get_family_size(df):
    df['Marital_Status'] = df['Marital_Status'].map(
        {'YOLO': 1, 'Absurd': 1, 'Alone': 1, 'Single': 1, 'Widow': 1, 'Divorced': 1, 'Together': 2, 'Married': 2})
    df['Family_Size'] = df.Marital_Status + df.Kidhome + df.Teenhome


def drop_redundants(df):
    df.drop('Dt_Customer', axis=1, inplace=True)
    df.drop('Z_CostContact', axis=1, inplace=True)
    df.drop('Z_Revenue', axis=1, inplace=True)
    df.drop('Marital_Status', axis=1, inplace=True)
    df.drop('ID', axis=1, inplace=True)
    df.drop('Year_Birth', axis=1, inplace=True)
    df.drop('AcceptedCmp1', axis=1, inplace=True)
    df.drop('AcceptedCmp2', axis=1, inplace=True)
    df.drop('AcceptedCmp3', axis=1, inplace=True)
    df.drop('AcceptedCmp4', axis=1, inplace=True)
    df.drop('AcceptedCmp5', axis=1, inplace=True)
    df.drop('Response', axis=1, inplace=True)
    df.drop('Kidhome', axis=1, inplace=True)
    df.drop('Teenhome', axis=1, inplace=True)


def preprocess_data(df):
    impute_missing_data(df)
    remove_outliers(df, 'Income')
    remove_outliers(df, 'Year_Birth')
    transform_age(df)
    trasform_education(df)
    combine_campaigns(df)
    get_family_size(df)
    drop_redundants(df)

    return df


def standardize_numericals(df):
    num_df = df[df.columns.difference(['Education', 'Complain', 'Campaigns_Accepted'])]
    num_df = num_df.apply(lambda x: ((x - x.mean()) / x.std()).round(2))
    return pd.concat(objs=[num_df, df['Education'], df['Complain'], df['Campaigns_Accepted']], axis=1)

# Data Visualizations


def histogram(df):
    df.hist()
    plt.show()


df = preprocess_data(raw)
stand_df = standardize_numericals(df)
print(stand_df)
# histogram(df)
