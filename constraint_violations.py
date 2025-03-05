'''
Analyses the CVs of all given results and prints a table including all provided results.
'''

import json
import pickle
import functions
import numpy as np
import pandas as pd
from random import choice
from collections import Counter
import matplotlib.pyplot as plt

files = str(input('List of filenames seperated by "," of results, or "[q]uit" to end: ')) 
l_files = files.replace(' ','').split(',')
if files in ['q','quit','']:
    exit()
net_ver_ = l_files[0].split('_')[2] # keep adaptable for different networks, provided the necessary data is available
if net_ver_ == 'HitG':
    net_ver = net_ver_+'_'+l_files[0].split('_')[3]
else:
    net_ver = net_ver_

with open('../dicts/Target_TFs_'+net_ver_+'.json','r') as f:
    Target_TFs = json.load(f)
with open('../dicts/TF_Targets_'+net_ver_+'.json','r') as f:
    TF_Targets = json.load(f)
with open('../dicts/non_optimizables_'+net_ver_+'.json','r') as f:
    non_opti = json.load(f)
mRNA_df = pd.read_pickle('../pickle/mRNA_collectri_'+net_ver+'_df.pkl')
Prot_time_df = pd.read_pickle('../pickle/Prot_time_collectri_'+net_ver+'_df.pkl')

with open('../dicts/colour_code_'+net_ver_+'_v2.json','r') as f:
    colour_code_a = json.load(f)
with open('../dicts/colour_code_PSites.json','r') as f:
    colour_code_b = json.load(f)

colours_alpha = [val for key,val in colour_code_a.items()] # the colour code as in the Thesis, for the TFs
colours_beta = [val for key,val in colour_code_b.items()] # the colour code as in the Thesis, for the PSites

df_for_beta_ = pd.DataFrame() # will include the sums for all beta sets
df_for_alpha_ = pd.DataFrame() # will include the sums for all alpha sets

for file in l_files:
    Method = file[len('res_'):file.find('_'+net_ver)]
    
    dictionary,variation = functions.which_dictionary(file,Method,net_ver)

    with open(dictionary+'_alpha_dict.json','r') as f:
        alpha_pos_dict = json.load(f)
    with open(dictionary+'_beta_dict.json','r') as f:
        beta_pos_dict = json.load(f)

    x = functions.declare_x(file,Method)

    column_a = {'Method': Method+'_'+variation}
    column_b = {'Method': Method+'_'+variation}
    # finds the sum of all alphas for each mRNA and the sum of all betas for each TF
    for R in mRNA_df.columns:
        column_a[R] = np.round(np.sum(x[alpha_pos_dict[R][0]:alpha_pos_dict[R][1]]),5)
        for TF in Target_TFs[R]:
            column_b[TF] = np.round(np.sum(x[beta_pos_dict[TF][0]:beta_pos_dict[TF][1]]),5)
    
    # extrema
    max_alpha = max(x[list(alpha_pos_dict.items())[0][1][0]:list(alpha_pos_dict.items())[-1][1][-1]])
    min_alpha = min(x[list(alpha_pos_dict.items())[0][1][0]:list(alpha_pos_dict.items())[-1][1][-1]])
    
    max_beta = max(x[list(beta_pos_dict.items())[0][1][0]:list(beta_pos_dict.items())[-1][1][-1]])
    min_beta = min(x[list(beta_pos_dict.items())[0][1][0]:list(beta_pos_dict.items())[-1][1][-1]])
    
    column_a['max. alpha'] = max_alpha
    column_a['min. alpha'] = min_alpha
    column_b['max. beta'] = max_beta
    column_b['min. beta'] = min_beta
    
    # add to the dataframe
    df_for_beta_ = df_for_beta_._append(column_b,ignore_index=True).drop_duplicates()
    df_for_alpha_ = df_for_alpha_._append(column_a,ignore_index=True)

# can of course be saved, just add the lines :)
df_for_alpha = df_for_alpha_.fillna(0).reindex(sorted(df_for_alpha_.columns),axis=1).set_index('Method').sort_index()
df_for_beta = df_for_beta_.set_index('Method').reindex(sorted(df_for_beta_),axis=1).sort_index()
print(df_for_alpha)
print(df_for_beta)