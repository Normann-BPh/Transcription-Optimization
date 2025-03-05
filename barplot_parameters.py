'''
Plots a BarPlot of all porvided solution and saves them automatically. Used for analyzing CVs.
Doesn't show them "plt.show()" is commented out by default.
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

with open('../dicts/colour_code_'+net_ver_+'_v2.json','r') as f:
    colour_code_a = json.load(f)
with open('../dicts/colour_code_PSites.json','r') as f:
    colour_code_b = json.load(f)

colours_alpha = [val for key,val in colour_code_a.items()]
colours_beta = [val for key,val in colour_code_b.items()]


for file in l_files:
    Method = file[len('res_'):file.find('_'+net_ver_)]
    
    dictionary_link, variation = functions.which_dictionary(file,Method,net_ver)
    
    with open(dictionary_link+'_alpha_dict.json','r') as f:
        alpha_pos_dict = json.load(f)
    with open(dictionary_link+'_beta_dict.json','r') as f:
        beta_pos_dict = json.load(f)
    
    x = functions.declare_x(file,Method)
        
    # Formatting and such
    df_for_alpha_ = pd.DataFrame()
    df_for_beta_ = pd.DataFrame()
    for R in mRNA_df.columns:
        row_a = {'GeneID':R}
        alpha = alpha_pos_dict[R][0]
        for TF in Target_TFs[R]:
            dex_a = Target_TFs[R].index(TF)
            row_a[TF] = x[alpha+dex_a]
            #----------------------#
            column_b = {'TF': TF}
            beta = beta_pos_dict[TF][0]
            dex_b = 0
            for P in Prot_time_df[TF]:
                if str(P) != 'nan':
                    column_b['P'+str(dex_b)] = x[beta+dex_b]
                    dex_b = dex_b + 1
                else:
                    continue
            df_for_beta_ = df_for_beta_._append(column_b,ignore_index=True).drop_duplicates()
        df_for_alpha_ = df_for_alpha_._append(row_a,ignore_index=True)
    df_for_alpha = df_for_alpha_.fillna(0).reindex(sorted(df_for_alpha_.columns),axis=1).set_index('GeneID').sort_index()
    df_for_alpha.to_csv('../Formatted_X_values_'+Method+'_'+variation+'_alpha.csv')
    df_for_beta = df_for_beta_.fillna(0).set_index('TF').sort_index()
    df_for_beta.to_csv('../Formatted_X_values_'+Method+'_'+variation+'_beta.csv')
    
    width = 0.8

    df = df_for_alpha
    
    ax = df.plot.bar(stacked=True,legend=False,color=colours_alpha,width=width, figsize=(23, 18))
    ax.set_ylabel('alpha value',fontsize='xx-large')
    #ax.set_title('Alpha values of the result from optimizing with '+Method+'_'+variation,fontsize='xx-large')
    ax.set_xlabel('mRNA',fontsize=20)
    
    plt.subplots_adjust(top=0.999,bottom=0.0999,left=0.05,right=0.999)
    plt.savefig('../plots/alpha_bar/alphas_'+file[4:]+'.png')
    plt.close()
    
    df = df_for_beta
    
    ax = df.plot.bar(stacked=True,legend=False,color=colours_beta,width=width, figsize=(23, 18))
    ax.set_ylabel('beta value',fontsize='xx-large')
    #ax.set_title('Beta values of the result from optimizing with '+Method+'_'+variation,fontsize='xx-large')
    ax.set_xlabel('TF',fontsize=20)
    
    plt.subplots_adjust(top=0.999,bottom=0.0999,left=0.05,right=0.999)
    plt.savefig('../plots/beta_bar/betas_'+file[4:]+'.png')
    plt.close()
#plt.show()