import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, date
from sklearn import cluster, preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

raw = pd.read_csv('Data\marketing_data.csv', delimiter='\t')

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


def remove_features(df, features: list):
    return (df[df.columns.difference(features)])


def remove_categorical(df):
    return df[df.columns.difference(['Education', 'Complain', 'Campaigns_Accepted'])]


def concat_features(non_concat_df, df_to_concat, features: list):
    non_concat_df.reset_index(drop=True, inplace=True)
    df_to_concat.reset_index(drop=True, inplace=True)
    for feature in features:
        non_cat_df = pd.concat(objs=[non_concat_df, df_to_concat[feature]], axis=1, copy=False)
    return non_cat_df


def concat_categorical(df, non_cat_df):
    df.reset_index(drop=True, inplace=True)
    non_cat_df.reset_index(drop=True, inplace=True)
    return pd.concat(objs=[num_df, df['Education'], df['Complain'], df['Campaigns_Accepted']], axis=1, copy=False)


# Data Visualizations and Transformations

def standardize(df):
    num_df = df.apply(lambda x: ((x - x.mean()) / x.std()).round(2))
    return num_df


def power_transform(df):
    pt = preprocessing.PowerTransformer(standardize=True, copy=False)
    pt_df = pt.fit_transform(df)
    return pd.DataFrame(pt_df, columns=pt.feature_names_in_)


def principal_component_analysis(df, components):
    pca = PCA(n_components=components)
    pca.fit(stand_nums)
    pca_df = pd.DataFrame(pca.transform(stand_nums))
    return (pca, pca_df)


def linear_discriminant_analysis(df, target, components, plot: bool):
    lda = LinearDiscriminantAnalysis(n_components=components)
    lda_df = pd.DataFrame(lda.fit_transform(X=df, y=target))
    if components == 2 and plot:
        scatter_2D(lda_df, target)
    elif components == 3 and plot:
        scatter_3D(lda_df, target)
    return lda_df


def histogram(df):
    df.hist()
    plt.show()


def correlation_heatmap(df):
    sns.heatmap(df.corr())
    plt.show()


def scatter_2D(df, target):
    Xax = df[0]
    Yax = df[1]
    labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
    cdict = {0: 'purple', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'orange', 5: 'red'}
    alpha = {0: .2, 1: .2, 2: .2, 3: .4, 4: .4, 5: .6}
    for l in np.unique(target):
        indxs = np.where(target == l)
        for ix in indxs:
            plt.scatter(Xax[ix], Yax[ix], c=cdict[l], label=labels[l], alpha=alpha[l])
    plt.legend()
    plt.show()


def scatter_3D(df, target):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    Xax = df[0]
    Yax = df[1]
    Zax = df[2]
    labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
    cdict = {0: 'purple', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'orange', 5: 'red'}
    alpha = {0: .2, 1: .2, 2: .2, 3: .4, 4: .4, 5: .6}
    for l in np.unique(target):
        indxs = np.where(target == l)
        for ix in indxs:
            ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], label=labels[l], alpha=alpha[l])
    plt.legend()
    plt.show()


def scree_plot(pca):
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    plt.ylabel('% of Explained Variance')
    plt.xlabel('Principal Components')
    plt.show()


def pca_eigen_vectors(pca_df, pca):
    features = ['Feature' + str(x) for x in range(1, len(pca_df.columns) + 1)]
    loading_scores = pd.DataFrame(pca.components_, index=features).apply(lambda x: abs(x))
    return loading_scores


def pca_eigen_values(pca):
    return pca.explained_variance_


def k_means(df, clusters):
    kmeans = cluster.KMeans(n_clusters=clusters, random_state=39, n_init='auto')
    kmeans.fit(df)
    dim = len(df.columns)
    if dim == 2:
        scatter_2D(df, kmeans.labels_)
    elif dim == 3:
        scatter_3D(df, kmeans.labels_)


def elbow_plot(df, iterations):
    SSE = []
    ks = []
    for k in range(1, iterations + 1):
        kmeans = cluster.KMeans(n_clusters=k, random_state=39, n_init='auto')
        kmeans.fit(df)
        SSE.append(kmeans.inertia_)
        ks.append(k)
    plt.plot(ks, SSE)
    plt.show()


df = preprocess_data(raw)
non_cat_df = remove_categorical(df)
stand_nums = standardize(non_cat_df)
stand_gauss_df = concat_features(power_transform(remove_features(non_cat_df, ['Recency'])), stand_nums, ['Recency'])
(pca, pca_df) = principal_component_analysis(stand_nums, 3)
lda_df = linear_discriminant_analysis(stand_gauss_df, df['Campaigns_Accepted'], 3, False)
