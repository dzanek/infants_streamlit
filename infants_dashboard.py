import streamlit as st
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib

import sklearn as sk
import numpy as np
import scipy.stats as ss

import os
st.set_page_config(layout="wide")
st.title('Infant Gut microbiome in Russia and Finland')

rus_ids = [i.strip() for i in open('id_rus.srx')]
fin_ids = [i.strip() for i in open('id_fin.srx')]
all_ids = rus_ids+fin_ids

country_map = {i:'Rosja' for i in rus_ids}
country_map.update({i:'Finlandia' for i in fin_ids})

samples_map = [i.strip().split() for i in open('samples.map')]
samples_map = {i[1]:i[0] for i in samples_map}

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
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms


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

def load_data(tax_level='P'):
    from sklearn.preprocessing import StandardScaler
    taxonomy_dataframe_P = pd.DataFrame()
    for i in all_ids:
        #print(taxonomy_dataframe_P)
        sample = pd.read_table(f'bracken_{tax_level}/{samples_map[i]}.bracken',index_col=0)
        sample = sample[['fraction_total_reads']]
        sample.columns = [i]
        if len(taxonomy_dataframe_P.columns) == 0: 
            print('....')
            taxonomy_dataframe_P = sample.copy()
        else:
            taxonomy_dataframe_P = pd.merge(taxonomy_dataframe_P, sample[[i]], how='outer', suffixes=("", ""),left_index=True, right_index=True)
    #taxonomy_dataframe_P['phylum'] = taxonomy_dataframe_P.index
    taxonomy_dataframe_P.fillna(0.0,inplace=True)
    taxonomy_dataframe_P.index.rename('phylum',inplace=True)
    taxonomy_dataframe_P['mean'] = taxonomy_dataframe_P.apply(lambda x: np.average(x),axis=1)
    taxonomy_dataframe_P.sort_values(by='mean',ascending=False,inplace=True)
    print(len(taxonomy_dataframe_P))
    taxonomy_dataframe_P.drop('mean',axis=1,inplace=True)
    taxonomy_dataframe_P = taxonomy_dataframe_P.loc[(taxonomy_dataframe_P == 0.0).mean(axis=1) < 0.9]
    print(len(taxonomy_dataframe_P))
    return taxonomy_dataframe_P

def scale_dataframe(taxonomy_dataframe_P):
    cols = taxonomy_dataframe_P.columns
    ind = taxonomy_dataframe_P.index
    scaler = StandardScaler()
    taxonomy_dataframe_P = scaler.fit_transform(taxonomy_dataframe_P) 
    taxonomy_dataframe_P = pd.DataFrame(taxonomy_dataframe_P)
    taxonomy_dataframe_P.columns = cols
    taxonomy_dataframe_P.index = ind
    return taxonomy_dataframe_P

def plot_barplot(data):
    fig, phylum_bar_kraken = plt.subplots()
    phylum_table_kraken = data
    #phylum_table_kraken
    phylum_table_kraken.T.sort_values(by=['Actinobacteria','Bacteroidetes'],ascending=[False,False]).plot(ax=phylum_bar_kraken, kind="bar", stacked=True)
    phylum_bar_kraken.legend(list(phylum_table_kraken.index)[:5],loc=1)
    phylum_bar_kraken.set_xticklabels([country_map[i] for i in phylum_table_kraken.columns])
    phylum_bar_kraken.set_xlabel('Kraj pochodzenia próbki')
    phylum_bar_kraken.set_ylabel('Proporcja danego typu bakterii w próbce')
    return fig

def plot_pca(data):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    pcagenus = PCA(n_components=2)
    pcagenus.fit(data)
    var1, var2 = pcagenus.explained_variance_ratio_
    fig, ax = plt.subplots()
    fig, ax = plt.subplots()
#    ax=fig.add_axes([0,0,1,1])
    ax.scatter(pcagenus.components_[0][len(rus_ids):], pcagenus.components_[1][len(rus_ids):],linewidths=2,edgecolors='#666666',s=300,color=sns.color_palette()[0],label='Probki Rosyjskie')
    ax.scatter(pcagenus.components_[0][:len(rus_ids)], pcagenus.components_[1][:len(rus_ids)],linewidths=2,edgecolors='#666666',s=300,color=sns.color_palette()[1],label='Probki Fińskie')

    confidence_ellipse(pcagenus.components_[0][len(rus_ids):], pcagenus.components_[1][len(rus_ids):],ax,n_std=3,edgecolor=sns.color_palette()[0])
    confidence_ellipse(pcagenus.components_[0][:len(rus_ids)], pcagenus.components_[1][:len(rus_ids)],ax,n_std=3,edgecolor=sns.color_palette()[1])

    ax.legend()
    ax.set_xlabel(f'PC1 {round(100*var1,2)}%')
    ax.set_ylabel(f'PC2 {round(100*var2,2)}%')
    plt.tight_layout()   
    return fig

def build_model(data):
    phylum_feature_table = data.T
    phylum_feature_table['location'] = phylum_feature_table.index.map(lambda x: 1 if country_map[x] == 'Rosja' else 0)
    
    from scipy import interp

    from sklearn.datasets import make_classification
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve
    from sklearn.metrics import plot_roc_curve
    from sklearn.metrics import auc
    
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
            label='Model losowy', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color=sns.color_palette()[0],
            label=r'Średnie ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 odch. standardowe')

    ax.set_xlabel('1 - specyficzność')
    ax.set_ylabel('czułość')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="")
    ax.legend(loc="lower right")

    return fig, classifier, features

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
    ax.set_xlabel('Pięć cech o najwyższym wkładzie w predykcję')
    #plt.ylim(39,)
    plt.tight_layout()   
    return fig
    


data_load_state = st.text('Loading data...')
data = load_data('P')
data_load_state.text("Done!")

st.subheader('Raw data')
st.write(data)

col1, col2 = st.columns(2)


col1.subheader('Taxonomy profile of the cohort')
col1.pyplot(plot_barplot(data))
col1.markdown('The stacked bar plot sorted by the most abundant taxa in the cohort.')

col1.subheader('PCA')
col1.pyplot(plot_pca(data))
col1.markdown('Principal Componen Analysis plot based on the taxonomy table.')

rocauc, classifier, features = build_model(data)

col2.subheader('Classification performance (ROC AUC)')
col2.pyplot(rocauc, classifier)
col2.markdown('ROC AUC plot from 10 times repeated, 5-fold Cross Validation, with Standard Deviation from repetitions marked as grey area.')

col2.subheader('Most important features')
col2.pyplot(get_top_features(classifier, features))
col2.markdown('The set of features most important for making predictions by trained model.')