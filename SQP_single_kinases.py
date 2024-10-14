import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from scipy.optimize import NonlinearConstraint
import sys
from datetime import datetime
import functions

from multiprocessing import Pool

with open("../dicts/Target_TFs_kinases.json", "r") as f:
    Target_TFs = json.load(f)
mRNA_df = pd.read_pickle("../pickle/mRNA_collectri_kinases_df.pkl")
with open("../dicts/TF_Targets_kinases.json", "r") as f:
    TF_Targets = json.load(f)
Prot_time_df = pd.read_pickle("../pickle/Prot_time_collectri_kinases_df.pkl")
with open("../dicts/non_optimizables_kinases.json", "r") as f:
    non_opti = json.load(f)

with open('../dicts/SQP/SQP_kinases_single_alpha_dict.json','r') as f:
    alpha_pos_dict = json.load(f)
with open('../dicts/SQP/SQP_kinases_single_beta_dict.json','r') as f:
    beta_pos_dict = json.load(f)
with open('../dicts/SQP/SQP_kinases_single_error_dict.json','r') as f:
    error_pos_dict = json.load(f)

d1 = datetime.today()
print(d1.strftime('%d-%m-%Y'))

last_f = list(error_pos_dict.items())[-1][-1][-1] + 9
n_var = list(beta_pos_dict.items())[-1][-1][-1]
print(last_f,n_var)
def error_sum(x):
    return sum(x[0:last_f])
def error_sum_grad(x):
    grad = np.zeros(n_var)
    grad[0:last_f] = 1
    return grad

boundsl = np.zeros(n_var) # lower bound for alpha and errors
boundsl[list(beta_pos_dict.items())[0][-1][0]:] = -1 # lower bound for beta 
boundsu = np.ones(n_var) # upper bound for alpha and beta
boundsu[0:last_f] = 200 # upper bound for errors
bounds = Bounds(boundsl,boundsu)

def SQP_fun_sin_new_new(x,error_pos_dict,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs):
    fun_res = np.array([])
    for R in mRNA_df.columns:
        begin_alpha, end_alpha = alpha_pos_dict[R]
        alpha = x[begin_alpha:end_alpha]
        # ---
        begin_fp,begin_fm = error_pos_dict[R]
        fp = x[begin_fp:begin_fp+9]
        fm = x[begin_fm:begin_fm+9]
        # ---
        unit_ = np.arange(len(Target_TFs[R]*9)).reshape(len(Target_TFs[R]),9)
        unit = np.zeros_like(unit_,dtype=float)
        for T in Target_TFs[R]:
            begin_beta, end_beta = beta_pos_dict[T]
            dex = Target_TFs[R].index(T)
            if end_beta - begin_beta > 1:
                P = np.sum([(x[(begin_beta+(list(Prot_time_df[T][1:]).index(site)))]*np.array(site)) for site in Prot_time_df[T][1:] if np.ndim(site) > 0],axis=0)
                unit[dex] = alpha[dex] * np.array(Prot_time_df[T][0]) * (x[begin_beta] + P)
            else:
                unit[dex] = alpha[dex] * np.array(Prot_time_df[T][0]) * x[begin_beta]
        unit = np.array(mRNA_df[R]) - np.sum(unit,axis=0) - fp + fm
        fun_res = np.concatenate((fun_res,unit))
    # ---
    for a in alpha_pos_dict:
        unit = 1 - sum(x[alpha_pos_dict[a][0]:alpha_pos_dict[a][1]])
        fun_res = np.append(fun_res,unit)
    # ---
    for a in beta_pos_dict:
        unit = 1 - sum(x[beta_pos_dict[a][0]:beta_pos_dict[a][1]])
        fun_res = np.append(fun_res,unit)
    return fun_res

'''
def SQP_jac_sin_new_new(x,error_pos_dict,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs,n_var):
    jac_res = np.zeros(n_var*len(mRNA_df.columns)*9).reshape(len(mRNA_df.columns)*9,n_var)
    for R in mRNA_df.columns:
        dex_ = list(mRNA_df.columns).index(R)*9
        # ---
        begin_fp, begin_fm = error_pos_dict[R]
        jac_res[dex_,begin_fp:begin_fp+9] = - 1
        jac_res[dex_,begin_fm:begin_fm+9] = + 1
        # ---
        begin_alpha, end_alpha = alpha_pos_dict[R]
        alpha = x[begin_alpha:end_alpha]
        # ---
        unit = np.zeros(len(Target_TFs[R]*9)).reshape(len(Target_TFs[R]),9)
        # ---
        for T in Target_TFs[R]:
            begin_beta, end_beta = beta_pos_dict[T]
            dex = Target_TFs[R].index(T)
            if end_beta - begin_beta > 1:
                P = np.sum([(x[(begin_beta+(list(Prot_time_df[T][1:]).index(site)))]*np.array(site)) for site in Prot_time_df[T][1:] if np.ndim(site) > 0],axis=0)
                unit[dex] = alpha[dex] * np.array(Prot_time_df[T][0]) * (x[begin_beta] + P)
                jac_res[dex_:dex_+9,begin_beta:end_beta] = np.insert((alpha[dex] * np.array(Prot_time_df[T][0]) * ([(x[(begin_beta+(list(Prot_time_df[T][1:]).index(site)))]*np.array(site)) for site in Prot_time_df[T][1:] if np.ndim(site) > 0])),0,(alpha[dex]*np.array(Prot_time_df[T][0])),axis=0).T
            else:
                unit[dex] = alpha[dex] * np.array(Prot_time_df[T][0]) * x[begin_beta]
                jac_res[dex_:dex_+9,begin_beta:end_beta] = (alpha[dex] * np.array(Prot_time_df[T][0])).reshape(9,1)
        jac_res[dex_:dex_+9,begin_alpha:end_alpha] = (unit/alpha[:,None]).T
        jac_res[dex_:dex_+9,begin_fp:begin_fp+9] = - 1
        jac_res[dex_:dex_+9,begin_fm:begin_fm+9] = + 1
    # ---
    for a in alpha_pos_dict:
        unit = np.zeros(n_var)
        unit[alpha_pos_dict[a][0]:alpha_pos_dict[a][1]] = - 1
        jac_res = np.concatenate((jac_res,[unit]),axis=0)
    # ---
    for b in beta_pos_dict:
        unit = np.zeros(n_var)
        unit[beta_pos_dict[b][0]:beta_pos_dict[b][1]] = - 1
        jac_res = np.concatenate((jac_res,[unit]),axis=0)
    return jac_res
'''

callback_ndarray = np.array([])

def callback_fun(x,step=[0]):
    global callback_ndarray
    callback_ndarray = np.append(callback_ndarray,x)
    print(step[0],error_sum(x))
    step[0] += 1

eq_cons = {'type': 'eq',
            'fun': lambda x: SQP_fun_sin_new_new(x,error_pos_dict,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs),
			#'jac': lambda x: SQP_jac_sin_new_new(x,error_pos_dict,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs,n_var)
			}
            
x0 = np.random.rand(n_var)
maxiter = 100
res = minimize(error_sum, x0, method='SLSQP', jac=error_sum_grad, 
               constraints=[eq_cons], options={'ftol':0.005, 'disp':True, 'maxiter':maxiter},
               bounds=bounds,callback=callback_fun)
print(res.x)
print(callback_ndarray.reshape(maxiter,n_var))

d2 = datetime.today()
print(d2.strftime('%d-%m-%Y'))
dif = d1 - d2
Time = dif.seconds + (dif.days * 3600 * 24)

d = {'x':res.x, 'x0':x0}
res_df = pd.DataFrame(data=d)
name = 'SQP_kinases_single_{maxiter}_{d1}_{Time}'.format(Time=Time,d1=d1.strftime('%d-%m-%Y'),maxiter=maxiter)
res_df.to_csv('./results/res_{name}.csv'.format(name=name))
np.save('./results/x0/x0_{name}.npy'.format(name=name),x0)
np.save('./results/cal_{name}.npy'.format(name=name),callback_ndarray.reshape(maxiter,n_var))