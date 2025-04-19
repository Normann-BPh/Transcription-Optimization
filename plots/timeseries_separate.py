'''
This script constructs plots of the provided data, you can decide through inputs how an open result should be presented.
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

mrna = [(mRNA_df[j][t]) for j in mRNA_df for t in range(9)]
t = [4,8,15,30,60,120,240,480,960]

colour_dict = {'DE_rmsd':"#8958d0",'DE_single':"#b5ab30",'GA_rmsd':"#c45393",
            'GA_single':"#6aa54f",'SQP_rmsd':"#c7593c",'SQP_single':"#26cdca"}

for file in l_files:
    Method = file[len('res_'):file.find('_'+net_ver)]
    
    dictionary,variation = functions.which_dictionary(file,Method,net_ver)

    with open(dictionary+'_alpha_dict.json','r') as f:
        alpha_pos_dict = json.load(f)
    with open(dictionary+'_beta_dict.json','r') as f:
        beta_pos_dict = json.load(f)

    x = functions.declare_x(file,Method)

    esti = functions.core_model_function(x,Method,Protein_Matrices_dict,beta_pos_dict,alpha_pos_dict,Target_TFs,Prot_time_df,mRNA_df)
    
    zeile = 0
    spalte = 0

    count = 0
    print('Currently open file: ', file)
    genes = input('Enter "all" to plot for every mRNA in one image (not plot), \n or a list of gene/mRNA Names to plot in indevidual files \n (use "spe" to plot for all genes/mRNAs separately): ')
    l_genes = genes.replace(' ','').split(',')
    
    colour = colour_dict[Method+'_'+variation]
    
    if genes == 'all' or genes == '':
        fig, axs = plt.subplots(int(size/5),5, figsize=(32,18)) # should be change depending on how many mRNAs need to be plotted
        for i in mRNA_df.columns:
            exp = mrna[count:count+9]
            cal = esti[count:count+9]
            
            axs[zeile,spalte].plot(t,cal, label=i+'_estimated')
            axs[zeile,spalte].plot(t,exp, label=i+'_measured')

            axs[zeile,spalte].set_xscale('function', functions=(forward,inverse))
            axs[zeile,spalte].set_xlim(0,1000)
            axs[zeile,spalte].xaxis.set_major_locator(FixedLocator(t))
            axs[zeile,spalte].grid(True)
            
            axs[zeile,spalte].set_xlabel('time in min')
            axs[zeile,spalte].set_ylabel('fold change')
            axs[zeile,spalte].legend(loc='upper right')            
            
            spalte += 1
            if spalte > 4:
                zeile += 1
                spalte = 0
            count += 9
        #fig.suptitle('Fold change over 16h of experimental data and estimation. Method: '+Method+'; Variation: '+variation+'.',fontsize='xx-large')
        plt.subplots_adjust(top=0.999,bottom=0.03)
        plt.savefig('../plots/combined/'+Method+file[4+len(Method):]+'_use_or_not.png')
        plt.show()
    elif genes == 'two':
        fig, axs = plt.subplots(int(size/10),5, figsize=(32,18))
        for i in list(mRNA_df.columns)[:int(size/2)]:
            exp = mrna[count:count+9]
            cal = esti[count:count+9]
            
            axs[zeile,spalte].plot(t,cal, label=i+'_H',linewidth=3,color=colour)
            axs[zeile,spalte].plot(t,exp, label=i+'_R',linewidth=3,color='black')

            axs[zeile,spalte].set_xscale('function', functions=(forward,inverse))
            axs[zeile,spalte].set_xlim(0,1000)
            axs[zeile,spalte].xaxis.set_major_locator(FixedLocator(t))
            axs[zeile,spalte].grid(True)
            
            axs[zeile,spalte].set_xlabel('time in min')
            axs[zeile,spalte].set_ylabel('fold change')
            axs[zeile,spalte].legend(loc='upper right',fontsize=18)            
            
            spalte += 1
            if spalte > 4:
                zeile += 1
                spalte = 0
            count += 9
        plt.subplots_adjust(top=0.999,bottom=0.03,left=0.05,right=0.999)
        plt.savefig('../plots/combined/'+Method+file[4+len(Method):]+'_p1.png')
        
        zeile = 0
        spalte = 0
        
        fig, axs = plt.subplots(int(size/10),5, figsize=(32,18))
        for i in list(mRNA_df.columns)[int(size/2):]:
            exp = mrna[count:count+9]
            cal = esti[count:count+9]
            
            axs[zeile,spalte].plot(t,cal, label=i+'_H',linewidth=3,color=colour)
            axs[zeile,spalte].plot(t,exp, label=i+'_R',linewidth=3,color='black')

            axs[zeile,spalte].set_xscale('function', functions=(forward,inverse))
            axs[zeile,spalte].set_xlim(0,1000)
            axs[zeile,spalte].xaxis.set_major_locator(FixedLocator(t))
            axs[zeile,spalte].grid(True)
            
            axs[zeile,spalte].set_xlabel('time in min')
            axs[zeile,spalte].set_ylabel('fold change')
            axs[zeile,spalte].legend(loc='upper right',fontsize=18)              
            
            spalte += 1
            if spalte > 4:
                zeile += 1
                spalte = 0
            count += 9
        plt.subplots_adjust(top=0.999,bottom=0.03,left=0.05,right=0.999)
        plt.savefig('../plots/combined/'+Method+file[4+len(Method):]+'_p2.png')
    else:
        if genes == 'spe':
            l_genes = mRNA_df.columns
        for gene in l_genes:
            fig, ax = plt.subplots(1,1, figsize=(32,18))
            count = list(mRNA_df.columns).index(gene)*9
            cal = esti[count:count+9]
            exp = mrna[count:count+9]
            
            ax.plot(t,cal, label=gene+'_estimated',linewidth=3)
            ax.plot(t,exp, label=gene+'_measured',linewidth=3)
            ax.grid(True)
            
            ax.set_xscale('function', functions=(forward,inverse))
            ax.set_xlim(0,1000)
            ax.xaxis.set_major_locator(FixedLocator(t))
            ax.grid(True)    
            
            ax.set_xlabel('time in min',fontsize='xx-large')
            ax.set_ylabel('fold change',fontsize='xx-large')
            ax.legend(loc='center right')
            
            fig.suptitle('Fold change over 16h of experimental data and estimation of '+gene+'. Method: '+Method+'; Variation: '+variation+'.',fontsize='xx-large')
            
            alpha = x[alpha_pos_dict[gene][0]:alpha_pos_dict[gene][1]]
            for T in Target_TFs[gene]:
                beta = x[beta_pos_dict[T][0]:beta_pos_dict[T][1]]
            
            plt.savefig('../plots/combined/separate/example_net/'+gene+'_'+Method+file[4+len(Method):]+'.png')
            if len(l_genes) == 1:
                plt.show()