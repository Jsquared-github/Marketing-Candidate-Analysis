import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, date
from sklearn import cluster, preprocessing
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Functions to clean data


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


def principal_component_analysis(df, target, components, plot: bool):
    pca = PCA(n_components=components)
    pca.fit(df)
    pca_df = pd.DataFrame(pca.transform(df))
    if components == 2 and plot:
        scatter_2D(pca_df, target)
    elif components == 3 and plot:
        scatter_3D(pca_df, target)
    return (pca, pca_df)


def linear_discriminant_analysis(df, target, components, plot: bool):
    lda = LinearDiscriminantAnalysis(n_components=components)
    lda_df = pd.DataFrame(lda.fit_transform(X=df, y=target))
    if components == 2 and plot:
        scatter_2D(lda_df, target)
    elif components == 3 and plot:
        scatter_3D(lda_df, target)
    return (lda, lda_df)


def tSNE(df, target, components, plot: bool):
    t_SNE = TSNE(n_components=components, perplexity=47.5, n_iter=1750, random_state=39)
    t_SNE_df = pd.DataFrame(t_SNE.fit_transform(df))
    if components == 2 and plot:
        scatter_2D(t_SNE_df, target)
    elif components == 3 and plot:
        scatter_3D(t_SNE_df, target)
    return (t_SNE, t_SNE_df)


def k_means(df, clusters, plot: bool):
    kmeans = cluster.KMeans(n_clusters=clusters, random_state=39, n_init='auto').fit(df)
    dim = len(df.columns)
    if dim == 2 and plot:
        scatter_2D(df, kmeans.labels_)
    elif dim == 3 and plot:
        scatter_3D(df, kmeans.labels_)
    return kmeans.labels_


def k_medoids(df, clusters, plot: bool):
    kmeds = KMedoids(n_clusters=clusters, init='random', random_state=39).fit(df)
    dim = len(df.columns)
    if dim == 2 and plot:
        scatter_2D(df, kmeds.labels_)
    elif dim == 3 and plot:
        scatter_3D(df, kmeds.labels_)
    return kmeds.labels_


def histogram(df):
    df.hist()
    plt.show()


def correlation_heatmap(df):
    sns.heatmap(df.corr())
    plt.show()


def scatter_2D(df, target):
    Xax = df[0]
    Yax = df[1]
    labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7'}
    cdict = {0: 'purple', 1: 'blue', 2: 'darkgreen', 3: 'yellow', 4: 'orange', 5: 'red', 6: 'pink', 7: 'brown'}
    alpha = {0: .2, 1: .2, 2: .4, 3: .4, 4: .4, 5: .6, 6: .6, 7: .8}
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
    labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7'}
    cdict = {0: 'purple', 1: 'blue', 2: 'darkgreen', 3: 'yellow', 4: 'orange', 5: 'red', 6: 'pink', 7: 'brown'}
    alpha = {0: .2, 1: .2, 2: .4, 3: .4, 4: .4, 5: .6, 6: .6, 7: .8}
    for l in np.unique(target):
        indxs = np.where(target == l)
        for ix in indxs:
            ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], label=labels[l], alpha=alpha[l])
    plt.legend()
    plt.show()


# Methods to Evaluate Transformations


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


def elbow_plot(df, iterations, method: str):
    SSE = []
    ks = []
    if method == 'kmeans':
        for k in range(2, iterations + 1):
            kmeans = cluster.KMeans(n_clusters=k, random_state=39, n_init='auto').fit(df)
            SSE.append(kmeans.inertia_)
            ks.append(k)
    elif method == 'kmedoids':
        for k in range(2, iterations + 1):
            kmeds = KMedoids(n_clusters=k, init='random', random_state=39).fit(df)
            SSE.append(kmeds.inertia_)
            ks.append(k)
    return ks, SSE


def silhouette_plot(df, iterations, method: str):
    sils = []
    ks = []
    if method == 'kmeans':
        for k in range(2, iterations + 1):
            kmeans = cluster.KMeans(n_clusters=k, random_state=39, n_init='auto').fit(df)
            sils.append(silhouette_score(df, kmeans.labels_))
            ks.append(k)
    elif method == 'kmedoids':
        for k in range(2, iterations + 1):
            kmeds = KMedoids(n_clusters=k, init='random', random_state=39).fit(df)
            sils.append(silhouette_score(df, kmeds.labels_))
            ks.append(k)
    return (ks, sils)


def optimal_k(df, iterations, method: str):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    (elb_ks, SSE) = elbow_plot(df, iterations, method)
    ax1.plot(elb_ks, SSE)
    ax1.set_title('Elbow Plot')
    (sil_ks, sils) = silhouette_plot(df, iterations, method)
    ax2.plot(sil_ks, sils)
    ax2.set_title('Silhouette Plot')
    plt.show()


def final_feature_plot(idx_df, series, val_df, title):
    final_df = pd.DataFrame()
    for c in np.unique(series):
        cluster_df = val_df.iloc[np.where(series == c)].mean()
        final_df[f'{c+1}'] = cluster_df
    print(final_df)
    final_df.T.plot.bar(subplots=True, layout=(4, 4), legend=False, rot=0)
    plt.suptitle(title)
    plt.show()


raw = pd.read_csv('Data\marketing_data.csv', delimiter='\t')
df = preprocess_data(raw)
non_cat_df = remove_categorical(df)
stand_nums_df = standardize(non_cat_df)
stand_gauss_df = concat_features(power_transform(remove_features(non_cat_df, ['Recency'])), stand_nums_df, ['Recency'])
non_cat_df['Campaigns_Accepted'] = df['Campaigns_Accepted']

(pca, pca_df) = principal_component_analysis(stand_nums_df, df['Campaigns_Accepted'], 3, False)
stand_nums_df['pca_kmean_cats'] = pd.DataFrame(k_means(pca_df, 4, False))
stand_nums_df['pca_kmed_cats'] = pd.DataFrame(k_medoids(pca_df, 4, False))

(lda, lda_df) = linear_discriminant_analysis(stand_gauss_df, df['Campaigns_Accepted'], 3, False)
stand_gauss_df['lda_kmean_cats'] = pd.DataFrame(k_means(lda_df, 6, False))
stand_gauss_df['lda_kmed_cats'] = pd.DataFrame(k_medoids(lda_df, 5, False))

# (tSNE, tSNE_df) = tSNE(stand_nums_df, df['Campaigns_Accepted'], 3, False)
# stand_nums_df['tSNE_kmean_cats'] = pd.DataFrame(k_means(tSNE_df, 5, False))
# stand_nums_df['tSNE_kmed_cats'] = pd.DataFrame(k_medoids(tSNE_df, 7, False))

final_feature_plot(stand_nums_df, stand_nums_df['pca_kmean_cats'], non_cat_df, title='PCA K-Means')
final_feature_plot(stand_nums_df, stand_nums_df['pca_kmed_cats'], non_cat_df, title='PCA K-Medoids')
final_feature_plot(stand_nums_df, stand_gauss_df['lda_kmean_cats'], non_cat_df, title='LDA K-Medoids')
final_feature_plot(stand_nums_df, stand_gauss_df['lda_kmed_cats'], non_cat_df, title='LDA K-Medoids')
# final_feature_plot(stand_nums_df, stand_nums_df['tSNE_kmean_cats'], non_cat_df, title='tSNE K-Medoids')
# final_feature_plot(stand_nums_df, stand_nums_df['tSNE_kmed_cats'], non_cat_df, title='tSNE K-Medoids')
