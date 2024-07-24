# Goals: 
# - Show how correlation and association could be used to improve student performance
# - Use modelling to draw conclusions and develop experiments
# - Work with multiple confounders to develop a complex model

# Questions: 
# - What influences low student performance within groups of people with sensitive characteristics?
# - What can be done to increase student performance within these groups?

# This scenario is pre-treatment causal analysis

# LIBRARIES

# Data Analysis
import pandas as pd
import numpy as np
import pytimetk as tk
from missingno import matrix

import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit
from sklearn.metrics import roc_auc_score

por_raw_df = pd.read_csv('data/student-por_raw.csv') 

por_df = por_raw_df.copy()    
por_df['grade_avg'] = por_df[['G1', 'G2', 'G3']].mean(axis=1)    
por_df['passed'] = [0 if g3 < 10 else 1 for g3 in por_df['G3']]

pass_fail_counts = por_df['passed'].value_counts()
total_students = len(por_df)
pass_fail_proportions = pass_fail_counts / total_students

por_df = por_df.drop(columns=['G1', 'G2', 'G3'])
por_binarized_df = por_df.binarize()
por_correlated_df = por_binarized_df.correlate(target="passed__0")

por_correlated_df.plot_correlation_funnel()

