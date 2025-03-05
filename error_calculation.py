'''
Calculates the MAE and RMSD of the provided solutions. Prints a table at the end. 
'''

import json
import pickle
import functions
import numpy as np
import pandas as pd
from random import choice
from collections import Counter

files = str(input('List of filenames seperated by "," of results, or "[q]uit" to end: '))
l_files = files.replace(' ','').split(',')
if files in ['q','quit','']:
    exit()
net_ver_ = l_files[0].split('_')[2]
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

Protein_Matrices_dict = dict(zip(list(Prot_time_df.columns),[np.array([np.array([site[time] for site in Prot_time_df[T][1:] if np.ndim(site) > 0]) for time in range(9)]) for T in Prot_time_df.columns]))

index_names = list(mRNA_df.columns)
column_names = []
mrna = [(mRNA_df[j][t]) for j in mRNA_df for t in range(9)]

MAE_ = np.array([])
RMSD_ = np.array([])
n_input = len(l_files)

for file in l_files:
    Method = file[len('res_'):file.find('_'+net_ver_)]
    
    dictionary_link, variation = functions.which_dictionary(file,Method,net_ver)
    
    column_names.append(Method+'_'+variation)
    
    with open(dictionary_link+'_alpha_dict.json','r') as f:
        alpha_pos_dict = json.load(f)
    with open(dictionary_link+'_beta_dict.json','r') as f:
        beta_pos_dict = json.load(f)
    
    x = functions.declare_x(file,Method)
    
    esti = functions.core_model_function(x,Method,Protein_Matrices_dict,beta_pos_dict,alpha_pos_dict,Target_TFs,Prot_time_df,mRNA_df)

    mae,rmsd = functions.MAE_RMSD(esti,mrna)

    MAE_ = np.append(MAE_,mae)
    RMSD_ = np.append(RMSD_,rmsd)

MAE = MAE_.reshape(n_input,len(mRNA_df.columns))
RMSD = RMSD_.reshape(n_input,len(mRNA_df.columns))

MAE_dict = dict(zip(column_names,np.round(MAE,5)))
RMSD_dict = dict(zip(column_names,np.round(RMSD,5)))

MAE_df = pd.DataFrame(data=MAE_dict,index=index_names)
RMSD_df = pd.DataFrame(data=RMSD_dict,index=index_names)

MAE_RMSD_Table = pd.concat([MAE_df,RMSD_df],axis=1)

MAE_RMSD_Table = MAE_RMSD_Table.sort_index()

MAE_RMSD_Table.loc['total'] = np.concatenate((np.sum(MAE,axis=1),np.sum(RMSD,axis=1)),axis=0)
MAE_RMSD_Table.loc['max'] = [np.max(MAE_RMSD_Table[i][:-1]) for i in MAE_RMSD_Table.columns]
MAE_RMSD_Table.loc['min'] = [np.min(MAE_RMSD_Table[i][:-2]) for i in MAE_RMSD_Table.columns]

print(MAE_RMSD_Table)