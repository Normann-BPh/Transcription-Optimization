'''
Collection of functions used throughout the optimization and evaluation process
'''

import json
import heapq
import pickle
import numpy as np
import pandas as pd
from random import choice
from collections import Counter
from pytensor import tensor as pt
from progress.bar import ChargingBar

## Core function of the model ##
def core_model_function(x,Method,Protein_Matrices_dict,beta_pos_dict,alpha_pos_dict,Target_TFs,Prot_time_df,mRNA_df):
    '''
    x: the array of size n_var
    Method: the used optimization Method, only relevant is MCMC, all other can be set to 'None' or the respective Name
    Protein_Matrices_dict: a dictionary containing the TF as key and the respective matrix of all Phosphosites (not including the Protein it self)
    beta_pos_dict: a dictionary with the TFs as keys and their first and last beta position in x as values
    alpha_pos_dict: a dictionary with the mRNAs as keys and their first and last alpha position in x as values
    Target_TFs: a dictionary of mRNAs as keys with a list of their TFs as values
    Prot_time_df: the dataframe containing the fold changes of at least the Protein/TF itself
    mRNA_df: the dataframe with all mRNAs and their fold changes
    '''
    F_res = np.array([])
    if Method == 'MCMC':
        for key,(val1,val2) in beta_pos_dict.items():
            beta_pos_dict[key] = [val1+216,val2+216]
    for R in mRNA_df.columns:
        begin_of_alpha,end_of_alpha = alpha_pos_dict[R]
        alpha = x[begin_of_alpha:end_of_alpha]
        # preparing the unit which is then filled step by step with the results per TF
        unit_ = np.arange(len(Target_TFs[R])*9).reshape(len(Target_TFs[R]),9) 
        unit = np.zeros_like(unit_,dtype=float)
        for T in Target_TFs[R]:
            begin_of_beta,end_of_beta = beta_pos_dict[T]
            if end_of_beta-begin_of_beta > 1: # i.e. one or more PSites exist
                TF_index = Target_TFs[R].index(T)
                P = np.sum(x[begin_of_beta+1:end_of_beta]*Protein_Matrices_dict[T],axis=1)
                unit[TF_index] = alpha[TF_index]* np.array(Prot_time_df[T][0]) * (x[begin_of_beta] + P)
            else:
                TF_index = Target_TFs[R].index(T)
                unit[TF_index] = alpha[TF_index] * np.array(Prot_time_df[T][0]) * x[begin_of_beta]
        # sum up over all TFs and add to the results array
        unit = np.sum(unit,axis=0) 
        F_res = np.append(F_res,unit)
    return F_res

## A function to automate calling the correct index-dictionary depending on the function you are about to analyse ##
def which_dictionary(file,Method,net_ver):
    '''
    file: the name of the optimization result file in the scheme: res_Method_kinases_variation_...
    Method: the method of optimization; MCMC, DE, GA or SQP
    returns the dictionary string i.e. '../dicts/Method/Method_kinases(_variation)' and variation/type of applied Method
    '''
    if Method == 'SQP' or Method == 'DE':
        Method_ = 'SQP'
        variation = 'rmsd'
        pos = file.find('rmsd')
        if pos == -1:
            variation = 'single'
            pos = file.find('single')
        dictionary = '../dicts/'+Method_+'/'+Method_+'_'+net_ver+'_'+file[pos:pos+len(variation)]
    else:
        variation = 'single'
        pos = file.find(variation)
        if pos == -1:
            variation = 'rmsd'
        dictionary = '../dicts/'+Method+'/'+Method+'_'+net_ver
    
    return dictionary,variation

## To easily transform the saved result to a usable ndarray for analysis ##
def declare_x(file,Method):
    '''
    file: the name of the optimization result file in the scheme: res_Method_kinases_variation_...
    Method: the method of optimization; MCMC, DE, GA or SQP
    returns the optimization results correctly formatted as an ndarray
    '''
    #-----------declare x_------------#
    if Method == 'DE' or Method == 'GA':
        x_ = np.load('../optimizations/results/for_BA/'+file+'.npy')
    elif Method == 'MCMC':
        with open('../optimizations/results/for_BA/est_parameters/alpha_'+file+'.json') as f:
            alpha_freq = json.load(f)
        with open('../optimizations/results/for_BA/est_parameters/beta_'+file+'.json') as f:
            beta_freq = json.load(f)
    else:
        x_ = pd.read_csv('../optimizations/results/for_BA/'+file+'.csv')
    #-----------declare x------------#
    if Method == 'GA':
        x = x_[99]
    elif Method == 'SQP':
        x = np.array(x_['x'])
    elif Method == 'MCMC':
        x = np.zeros(366)
        for i in alpha_freq:
            n = int(i[i.find('[')+1:i.find(']')])
            x[n] = alpha_freq[i][0]
        for i in beta_freq:
            n = int(i[i.find('[')+1:i.find(']')])
            x[n] = beta_freq[i][0]
    else:
        x = x_
    
    return x

## Function for mean absolute and root mean squared deviation ##
def MAE_RMSD(esti,mrna):
    '''
    mean absolute error of estimation to measurement
    root mean square deviation of estimation to measurement
    esti the ndarray containing the estimated values for all time points in order
    mrna the ndarray containing the measured values for all time point in order
    '''
    count = 0
    mae = np.array([])
    rmsd = np.array([])
    for i in range(int(len(esti)/9)):
        mae = np.append(mae,sum(np.abs(esti[count:count+9]-mrna[count:count+9]))/9)
        rmsd = np.append(rmsd,np.sqrt(np.sum(np.square(esti[count:count+9]-mrna[count:count+9]))/9))
        count +=9 # because we have nine time points
    return mae,rmsd

## rounds halfs; might not be needed ##
def rounding_halfs(arr):
    arr = np.round(arr*2)/2
    return arr