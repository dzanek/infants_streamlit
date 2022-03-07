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
SRR4408087
SRR4408201
SRR4408082
SRR4408085
SRR4408049
SRR4408188
SRR4408152
SRR4305204
SRR4408203
SRR4408123
SRR4305334
SRR4305376
SRR4408149
SRR4408059
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
SRR4305488
SRR4305368
SRR4305407
SRR4305409
SRR4305356
SRR4305286
SRR4305515
SRR4305483
SRR4305414
SRR4305508
SRR4305032
SRR4305361
SRR4305408
SRR4305285
SRR4305175
SRR4305396'''.strip())
    samples_b = samples_b.split()
    print(samples_b)
#st.success('Treatment loaded!')

group_map = {i:'Control' for i in samples_a}
group_map.update({i:'Treatment' for i in samples_b})

data = get_data_table(samples_a,samples_b)
st.subheader('Raw data')
st.write(data)

st.subheader('Taxonomy profile of the cohort')
st.pyplot(plot_barplot(data.iloc[:-1],group_map))
st.markdown('The stacked bar plot sorted by the most abundant taxa in the cohort.')

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
