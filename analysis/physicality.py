'''
Classifies the Sub_Sums, i.e. the fold-change values attributed to each TF, by their effect and impact.
'''

import json
import pickle
import functions
import numpy as np
import pandas as pd
from random import choice
from collections import Counter
import matplotlib.pyplot as plt

df_Coll = pd.read_csv('../../../tabels/CollecTRI.csv')
target_=df_Coll['target_genesymbol']
source_=df_Coll['source_genesymbol']

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
mRNA_df = pd.read_pickle('../pickle/mRNA_collectri_'+net_ver+'_df.pkl')
Prot_time_df = pd.read_pickle('../pickle/Prot_time_collectri_'+net_ver+'_df.pkl')



Protein_Matrices_dict = dict(zip(list(Prot_time_df.columns),[np.array([np.array([site[time] for site in Prot_time_df[T][1:] if np.ndim(site) > 0]) for time in range(9)]) for T in Prot_time_df.columns]))


def forward(x):
    return x**0.5

def inverse(x):
    return x**2

def TF_effect(R,T,x):
    begin_of_alpha,end_of_alpha = alpha_pos_dict[R]
    alpha = x[begin_of_alpha:end_of_alpha]
    begin_of_beta,end_of_beta = beta_pos_dict[T]
    dex = Target_TFs[R].index(T)
    if end_of_beta-begin_of_beta > 1:
        P = np.sum(x[begin_of_beta+1:end_of_beta]*Protein_Matrices_dict[T],axis=1)
        Sub_Sum = alpha[dex]* np.array(Prot_time_df[T][0]) * (x[begin_of_beta] + P)
    else:
        Sub_Sum = alpha[dex] * np.array(Prot_time_df[T][0]) * x[begin_of_beta]
    return effect

def PSite_effect(T,P,dex,x):
    begin_of_beta,end_of_beta = beta_pos_dict[T]
    pdex = begin_of_beta+dex
    Sub_Sum = np.array(P) * x[pdex]
    return Sub_Sum

for file in l_files:
    c_true = 0
    c_false = 0
    c_zero = 0
    Method = file[len('res_'):file.find('_'+net_ver_)]
    
    dictionary_link, variation = functions.which_dictionary(file,Method,net_ver)
    
    with open(dictionary_link+'_alpha_dict.json','r') as f:
        alpha_pos_dict = json.load(f)
    with open(dictionary_link+'_beta_dict.json','r') as f:
        beta_pos_dict = json.load(f)
    
    x = functions.declare_x(file,Method)
    print(x.shape)
    df_for_alpha_ = pd.DataFrame()
    df_for_verified_ = pd.DataFrame()
    df_for_beta_ = pd.DataFrame()
    for R in mRNA_df.columns:        
        row_a = {'GeneID':R}
        row_v = {'GeneID':R}
        for TF in Target_TFs[R]:
            verified = df_Coll['is_stimulation'][list(source_[(source_ == TF)&(target_ == R)].index)[0]]
            if verified == 1:
                verified = 'A' # activate
            else:
                verified = 'I' # inhibit
            column_b = {'TF': TF}
            Sub_Sum_Mean = np.mean(TF_effect(R,TF,x))
            ## Below the classification used for the Colloqium; defines two cases, 
            ## effect (inhibtion, activation or nothing) and
            ## impact (normal, repressed or enhanced)
            if Sub_Sum_Mean == 0:
                cat = '0'
            elif Sub_Sum_Mean < 0:
                print(TF,'-->',R,':',Sub_Sum_Mean,'|',verified) # just to see the rare inhibitions and compare them to CollecTRI
                cat = 'I' # inhibit
            else:
                cat = 'A' # activate
            if np.abs(Sub_Sum_Mean) <= np.abs(np.mean(Prot_time_df[TF][0])):
                if np.abs(Sub_Sum_Mean) == np.abs(np.mean(Prot_time_df[TF][0])):
                    row_a[TF] = 'n'+cat # normal
                else:
                    row_a[TF] = 'r'+cat # repressed
            else:
                row_a[TF] = 'e'+cat # enhanced
            
            ##  Below the classification used in the Thesis ##
            # elif Sub_Sum_Mean <= np.mean(Prot_time_df[TF][0]):
                # if Sub_Sum_Mean == np.mean(Prot_time_df[TF][0]):
                    # row_a[TF] = 'X'
                # else:
                    # print(TF,Sub_Sum_Mean)
                    # row_a[TF] = 'I'
            # else:
                # row_a[TF] = 'A'
            ## section below compares to the CollecTRI table
            if row_a[TF][1] == verified:
                row_v[TF] = True
                c_true += 1
            else:
                if row_a[TF][1] == '0':
                    row_v[TF] = None
                    c_zero += 1
                else:
                    row_v[TF] = False
                    c_false += 1
            dex = 1
            for P in Prot_time_df[TF][1:]:
                if str(P) != 'nan':
                    effect = np.mean(PSite_effect(TF,P,dex,x))
                    if effect == 0:
                            column_b['P'+str(dex)] = '0'
                    elif effect < np.mean(P):
                        if effect < np.mean(P):
                            column_b['P'+str(dex)] = 'I'
                    else:
                        if effect > np.mean(P):
                            column_b['P'+str(dex)] = 'A'
                        elif effect == np.mean(P):
                            column_b['P'+str(dex)] = 'X'
                    dex +=1
                else:
                    continue
            df_for_beta_ = df_for_beta_._append(column_b,ignore_index=True).drop_duplicates()
        df_for_alpha_ = df_for_alpha_._append(row_a,ignore_index=True)
        df_for_verified_ = df_for_verified_._append(row_v,ignore_index=True)
    df_for_alpha = df_for_alpha_.fillna('-').reindex(sorted(df_for_alpha_.columns),axis=1).set_index('GeneID').sort_index()
    df_for_verified = df_for_verified_.fillna('-').reindex(sorted(df_for_verified_.columns),axis=1).set_index('GeneID').sort_index()
    df_for_beta = df_for_beta_.fillna('-').set_index('TF').sort_index()
    print(df_for_alpha)
    print(df_for_verified)
    print(df_for_beta)
    df_for_alpha.to_csv('../Effect_enhanced_'+Method+'_'+variation+'_alpha.csv')
    df_for_verified.to_csv('../Comparison_enhanced_'+Method+'_'+variation+'_alpha.csv')
    df_for_beta.to_csv('../Effect_enhanced_'+Method+'_'+variation+'_beta.csv')
    print(c_false,c_true,c_zero)