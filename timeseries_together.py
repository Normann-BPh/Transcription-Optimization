'''
This should plot all provide results for every mRNA into the same plot.
'''

import json
import pickle
import functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,NullFormatter

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

size = len(mRNA_df.columns)

Protein_Matrices_dict = dict(zip(list(Prot_time_df.columns),[np.array([np.array([site[time] for site in Prot_time_df[T][1:] if np.ndim(site) > 0]) for time in range(9)]) for T in Prot_time_df.columns]))

def forward(x):
    return x**0.5

def inverse(x):
    return x**2

colour_dict = {'DE_rmsd':"#8958d0",'DE_single':"#b5ab30",'GA_rmsd':"#c45393",
            'GA_single':"#6aa54f",'SQP_rmsd':"#c7593c",'SQP_single':"#26cdca"}

mrna = [(mRNA_df[j][t]) for j in mRNA_df for t in range(9)]
t = [4,8,15,30,60,120,240,480,960]

for R in mRNA_df.columns:
    fig, ax = plt.subplots(1,1, figsize=(16,9))
    pos = list(mRNA_df.columns).index(R)*9
    exp = (mrna[pos:pos+9])
    
    for file in l_files:
        Method = file[len('res_'):file.find('_'+net_ver)]
        
        dictionary,variation = functions.which_dictionary(file,Method,net_ver)

        with open(dictionary+'_alpha_dict.json','r') as f:
            alpha_pos_dict = json.load(f)
        with open(dictionary+'_beta_dict.json','r') as f:
            beta_pos_dict = json.load(f)

        x = functions.declare_x(file,Method)

        esti = functions.core_model_function(x,Method,Protein_Matrices_dict,beta_pos_dict,alpha_pos_dict,Target_TFs,Prot_time_df,mRNA_df)
        
        cal = (esti[pos:pos+9])
        
        ax.plot(t,cal, label=Method+'_'+variation, linewidth=3, color = colour_dict[Method+'_'+variation])
    
    ax.plot(t,exp, label='measured', linewidth=3, color = 'black')
    ax.grid(True)

    ax.set_xscale('function', functions=(forward,inverse))
    ax.set_xlim(4,1000)
    ax.xaxis.set_major_locator(FixedLocator(t))

    ax.set_xlabel('time in min',fontsize='xx-large')
    ax.set_ylabel('fold change',fontsize='xx-large')
    ax.legend(loc='upper right',fontsize='xx-large')
    
    plt.subplots_adjust(top=0.999,bottom=0.0999,left=0.05,right=0.999)
    plt.savefig('../plots/combined/separate/all_in_one/'+R+'.png')
    
    plt.close()