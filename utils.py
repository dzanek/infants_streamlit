import os
import tqdm

import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import seaborn as sns 

import streamlit as st

#@st.cache
def get_xml(run_id):
    if not os.path.isfile(f'./cache/{run_id}.xml'):
        url = f'https://trace.ncbi.nlm.nih.gov/Traces/sra/?run={run_id}&retmode=xml'
        req = requests.get(url)
        print(url)
        with open(f'./cache/{run_id}.xml','wb') as dump:
            dump.write(req.content)
#        xml = req.content
#    else:
    xml = open(f'./cache/{run_id}.xml','rb')
    Bs_data = BeautifulSoup(xml, "xml")
    return Bs_data

def order_df(df,ascending=False):
    df['mean'] = df.apply(lambda x: np.average(x), axis=1)
    df.sort_values(by='mean',ascending=ascending,inplace=True)
    return df[df.columns[:-1]]


def build_taxonomy_table(samples,target,rank):
    ''' docstring place '''

    taxonomy_table = pd.DataFrame()
    for run_id in tqdm.tqdm(samples):
        print(f'Building based on {run_id}')
        run_df = pd.DataFrame()
        Bs_data = get_xml(run_id)
        for i in Bs_data.find('RUN').find('tax_analysis').find_all('taxon', {'rank':rank}):
            if i.get('rank') != rank:
                continue
            #print(f"{i.get('name')},{i.get('rank')},{i.get('total_count')}")
            tax_df = pd.DataFrame({run_id: float(i.get('total_count'))}, index=[i.get('name')])
            run_df = run_df.append(tax_df)
        taxonomy_table = pd.merge(left=taxonomy_table, right=run_df, how='outer', left_index=True, right_index=True)
    taxonomy_table = taxonomy_table/taxonomy_table.sum()
    #taxonomy_table = order_df(taxonomy_table)
    taxonomy_table.loc['target'] = target    
    return taxonomy_table

@st.cache
def get_data_table(samples_a=["SRR15021134","SRR15021145"], samples_b=["SRR15021131","SRR15021132"], rank='phylum'):
    ''' build data table from two lists of sra accesions '''
    if not os.path.isdir('./cache'):
        os.mkdir('./cache')
    print('Start building taxonomy table')
    taxonomy_table = pd.merge(left=build_taxonomy_table(samples_a,0,rank=rank),
                              right=build_taxonomy_table(samples_b,1,rank=rank),
                              how='outer', left_index=True, right_index=True)
    print('Done building taxonomy table')
    taxonomy_table.fillna(0.0,inplace=True)
    taxonomy_table = taxonomy_table.loc[(taxonomy_table == 0).mean(axis=1) < 0.9]
    ordered_table = order_df(taxonomy_table.loc[[i for i in taxonomy_table.index if i!= 'target']])
    ordered_table.loc['target'] = taxonomy_table.loc['target']
    return ordered_table


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



@st.cache
def scale_dataframe(taxonomy_dataframe_P):
    cols = taxonomy_dataframe_P.columns
    ind = taxonomy_dataframe_P.index
    scaler = StandardScaler()
    taxonomy_dataframe_P = scaler.fit_transform(taxonomy_dataframe_P)
    taxonomy_dataframe_P = pd.DataFrame(taxonomy_dataframe_P)
    taxonomy_dataframe_P.columns = cols
    taxonomy_dataframe_P.index = ind
    return taxonomy_dataframe_P

@st.cache
def plot_barplot(data,group_map):
    fig, taxonomy_bar = plt.subplots()
    print('will plot')
    #phylum_table_kraken = data
    #phylum_table_kraken
    #data = data.T.sort_values(by=['Actinobacteria','Bacteroidetes'],ascending=[False,False]).T
    data.T.plot(ax=taxonomy_bar, kind="bar", stacked=True)
    print('plotted!')
    taxonomy_bar.legend(list(data.index)[:5],loc=1)
    taxonomy_bar.set_xticklabels([group_map[i] for i in data.columns])
    taxonomy_bar.set_xlabel('Samples')
    taxonomy_bar.set_ylabel('Fraction of taxa in the system')
    return fig

def box_taxa(data,to_plot):
    fig, taxonomy_box = plt.subplots()
    sns.boxplot(data=data.T, x='target', y=to_plot)
    return fig
#@st.cache
def plot_pca(data,samples_a):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    pcagenus = PCA(n_components=2)
    pcagenus.fit(data)
    var1, var2 = pcagenus.explained_variance_ratio_
    fig, ax = plt.subplots()
    fig, ax = plt.subplots()
#    ax=fig.add_axes([0,0,1,1])
    ax.scatter(pcagenus.components_[0][samples_a:], pcagenus.components_[1][samples_a:],linewidths=2,edgecolors='#666666',s=300,color=sns.color_palette()[0],label='Treatment')
    ax.scatter(pcagenus.components_[0][:samples_a], pcagenus.components_[1][:samples_a],linewidths=2,edgecolors='#666666',s=300,color=sns.color_palette()[1],label='Control')

    confidence_ellipse(pcagenus.components_[0][samples_a:], pcagenus.components_[1][samples_a:],ax,n_std=3,edgecolor=sns.color_palette()[0])
    confidence_ellipse(pcagenus.components_[0][:samples_a], pcagenus.components_[1][:samples_a],ax,n_std=3,edgecolor=sns.color_palette()[1])

    ax.legend()
    ax.set_xlabel(f'PC1 {round(100*var1,2)}%')
    ax.set_ylabel(f'PC2 {round(100*var2,2)}%')
    plt.tight_layout()
    return fig

@st.cache
def build_model(data,group_map):
    from scipy import interp
    from sklearn.datasets import make_classification
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve
    from sklearn.metrics import plot_roc_curve
    from sklearn.metrics import auc

    phylum_feature_table = data.T
    #phylum_feature_table['location'] = phylum_feature_table.index.map(lambda x: 1 if group_map[x] == 'Rosja' else 0)
    
    features = list(phylum_feature_table.columns[:-1])
    target = phylum_feature_table.columns[-1]
    X, y = np.array(phylum_feature_table[features]), np.array(phylum_feature_table[target])
    #print(X[[3,5,7]])
    #print(y.shape)

    cv = RepeatedStratifiedKFold(n_splits=5,n_repeats=10)
    classifier = LogisticRegression()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        #print(y[test])
        viz = plot_roc_curve(classifier, X[test], y[test],
                             label='_nolegend_',
                             alpha=0.0, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)


    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color=sns.color_palette()[1],
            label='Random', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color=sns.color_palette()[0],
            label=r'Average ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. deviation')

    ax.set_xlabel('1 - specificity')
    ax.set_ylabel('sensitivity')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="")
    ax.legend(loc="lower right")

    return fig, classifier, features

#@st.cache
def get_top_features(classifier, features):

    feature_importance = abs(classifier.coef_[0])
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(pos[-5:], feature_importance[sorted_idx][-5:], align='center', )
    ax.set_yticks(pos[-5:])
    ax.set_yticklabels(np.array(features)[sorted_idx][-5:], fontsize=16)
    ax.set_xlabel('Top 5 features')
    #plt.ylim(39,)
    plt.tight_layout()
    return fig