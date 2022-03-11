import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import sklearn as sk
import numpy as np
import scipy.stats as ss

import os

from utils import *

#st.set_page_config(layout="wide")
st.title('SRA study based microbial signature discovery tool')
st.write('Sample data from Vatanen et al. 2016')
rus_ids = [i.strip() for i in open('id_rus.srx')]
fin_ids = [i.strip() for i in open('id_fin.srx')]
all_ids = rus_ids+fin_ids



#samples_map = [i.strip().split() for i in open('samples.map')]
#samples_map = {i[1]:i[0] for i in samples_map}


#data_load_state = st.text('Loading data...')
#data = load_data(all_ids,samples_map,'P')
#data_load_state.text("Done!")



col1, col2 = st.columns(2)

col1.subheader('Input control samples')

with st.spinner("Loading Control Samples"):
    samples_a = col1.text_area('SRA run ID (newline each)', '''SRR4305340
SRR4305202
SRR4408106
SRR4305207
SRR15021134
         '''.strip())
    samples_a = samples_a.split()
    print(samples_a)
#st.success('Control loaded!')

with st.spinner("Loading Treatment Samples"):
    col2.subheader('Input treatment samples')
    samples_b = col2.text_area('SRA run ID (newline each)', '''SRR4305272
SRR4305403
SRR4305496
SRR4305406
SRR15021131
'''.strip())
    samples_b = samples_b.split()
    print(samples_b)
print('Treatment loaded!')

tax_rank = st.selectbox('Select taxonomy rank to work with', ('phylum','class','order','family','genus','species'))

group_map = {i:'Control' for i in samples_a}
group_map.update({i:'Treatment' for i in samples_b})
print('groups mapped')
data = get_data_table(samples_a,samples_b,tax_rank)
print('got data table')
st.subheader('Raw data')
st.write(data.loc[data.mean(axis=1) > 0.01])
print('after DF print')

st.subheader('Taxonomy profile of the cohort')
st.pyplot(plot_barplot(data.iloc[:-1],group_map))
st.markdown('The stacked bar plot sorted by the most abundant taxa in the cohort.')


to_plot = st.selectbox('Choose taxa to inspect', data.index, )
st.pyplot(box_taxa(data,to_plot)) #sns.boxplot(data=data.T, x='target',y=to_plot)
s,p = ss.mannwhitneyu(data.T.loc[data.T['target'] == 0][to_plot],
                   data.T.loc[data.T['target'] == 1][to_plot])
st.write(f'p-value: {p}')


st.subheader('PCA')
st.pyplot(plot_pca(data.iloc[:-1], len(samples_a)))
st.markdown('Principal Componen Analysis plot based on the taxonomy table.')

rocauc, classifier, features = build_model(data,group_map)

st.subheader('Classification performance (ROC AUC)')
st.pyplot(rocauc, classifier)
st.markdown('ROC AUC plot from 10 times repeated, 5-fold Cross Validation, with Standard Deviation from repetitions marked as grey area.')

st.subheader('Most important features')
st.pyplot(get_top_features(classifier, features))
st.markdown('The set of features most important for making predictions by trained model.')
