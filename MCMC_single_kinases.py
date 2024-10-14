import sys
import json
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
from pytensor import tensor as pt
from datetime import datetime

print('*** Start script ***')
print(f'{pm.__name__}: v. {pm.__version__}')

with open("../dicts/Target_TFs_kinases.json", "r") as f:
    Target_TFs = json.load(f)
mRNA_df = pd.read_pickle("../pickle/mRNA_collectri_kinases_df.pkl")
with open("../dicts/TF_Targets_kinases.json", "r") as f:
    TF_Targets = json.load(f)
Prot_time_df = pd.read_pickle("../pickle/Prot_time_collectri_kinases_df.pkl")
with open("../dicts/non_optimizables_kinases.json", "r") as f:
    non_opti = json.load(f)

with open('../dicts/MCMC/MCMC_kinases_alpha_dict.json','r') as f:
    alpha_pos_dict = json.load(f)
with open('../dicts/MCMC/MCMC_kinases_beta_dict.json','r') as f:
    beta_pos_dict = json.load(f)

d1 = datetime.today()
print(d1.strftime('%d-%m-%Y'))

def MCMC_mu_sin_new(alpha,beta,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs):
    res = pt.zeros((0,), dtype='float64')
    for R in mRNA_df.columns:
        begin_of_alpha,end_of_alpha = alpha_pos_dict[R]
        a = alpha[begin_of_alpha:end_of_alpha]
        unit_shape = (len(Target_TFs[R]),9)
        unit = pt.zeros(unit_shape)
        for T in Target_TFs[R]:
            begin_of_beta,end_of_beta = beta_pos_dict[T]
            if end_of_beta-begin_of_beta > 1:
                dex = Target_TFs[R].index(T)
                P = pt.sum([(beta[(begin_of_beta+(list(Prot_time_df[T][1:]).index(site)))]*pt.as_tensor(site)) for site in Prot_time_df[T][1:] if np.ndim(site) > 0],axis=0)
                unit = pt.subtensor.set_subtensor(unit[dex],pt.as_tensor(a[dex]* pt.as_tensor(Prot_time_df[T][0]) * (beta[begin_of_beta] + P)))
            else:
                dex = Target_TFs[R].index(T)
                unit = pt.subtensor.set_subtensor(unit[dex],pt.as_tensor(a[dex] * pt.as_tensor(Prot_time_df[T][0]) * beta[begin_of_beta]))
        unit = pt.sum(unit,axis=0)
        res = pt.concatenate([res,unit],axis=0)
    return res

def MCMC_con(alpha,beta,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs):
    unit = np.array([])
    ones = pt.ones((len(alpha_pos_dict)+len(beta_pos_dict),))
    dex = 0
    for k in alpha_pos_dict:
        # ------------------------------------------------------
        unit = np.append(unit,pt.sum(alpha[alpha_pos_dict[k][0]:alpha_pos_dict[k][1]]))
    for l in beta_pos_dict:
        #  ------------------------------------------------------
        unit = 1 - pt.sum(beta[beta_pos_dict[l][0]:beta_pos_dict[l][1]])
    res = ones - unit
    return res


if __name__ == '__main__':
    Y_obs = pt.as_tensor(np.concatenate((np.array([(mRNA_df[j][t]) for j in mRNA_df for t in range(9)]),np.zeros(len(alpha_pos_dict)+len(beta_pos_dict)))))
    print(Y_obs)

    last_a = list(alpha_pos_dict.items())[-1][-1][-1]
    print(last_a)
    last_b = list(beta_pos_dict.items())[-1][-1][-1]
    print(last_b)
    
    with pm.Model() as RNA_Model:
        # Priors for unknown model parameters
        alpha = pm.Uniform("alpha", lower=0, upper=1, shape=(last_a,))
        beta = pm.Uniform("beta", lower=-1, upper=1, shape=(last_b,))

        mu = pt.concatenate([MCMC_mu_sin_new(alpha,beta,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs),MCMC_con(alpha,beta,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs)],axis=0)
        
        sigma = Y_obs - mu
        
        # Likelihood of observations
        Y = pm.Normal("Y", mu=mu, sigma=sigma, observed=Y_obs)
        
        # draw 1000 posterior samples
        idata = pm.sample(nuts={'target_accept':1,'max_treedepth':30},draws=10, cores=2)
    
    d2 = datetime.today()
    dif = d1 - d2
    Time = dif.seconds + (dif.days * 3600 * 24)
    
    az.summary(idata)
    az.to_json(idata,'./results/idata_MCMC_single_kinases_w_con_{d1}_{Time}.json'.format(Time=Time,d1=d1.strftime('%Y-%m-%d')))
    data_df = idata.to_dataframe()
    data_df.to_csv('./results/idata_MCMC_single_kinases_w_con_{d1}_{Time}.csv'.format(Time=Time,d1=d1.strftime('%Y-%m-%d')))