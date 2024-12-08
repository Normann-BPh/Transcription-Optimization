import sys
import json
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
from datetime import datetime
from pytensor import tensor as pt

print('*** Start script ***')
print(f'{pm.__name__}: v. {pm.__version__}')

with open("../dicts/Target_TFs_HitG.json", "r") as f:
    Target_TFs = json.load(f)
with open("../dicts/TF_Targets_HitG.json", "r") as f:
    TF_Targets = json.load(f)
with open("../dicts/non_optimizables_HitG.json", "r") as f:
    non_opti = json.load(f)

mRNA_df = pd.read_pickle("../pickle/mRNA_collectri_HitG_50_df.pkl")
Prot_time_df = pd.read_pickle("../pickle/Prot_time_collectri_HitG_df.pkl")

with open('../dicts/MCMC/MCMC_HitG_50_alpha_dict.json','r') as f:
    alpha_pos_dict = json.load(f)
with open('../dicts/MCMC/MCMC_HitG_50_beta_dict.json','r') as f:
    beta_pos_dict = json.load(f)

the_prot_dict = dict(zip(list(Prot_time_df.columns),[np.array([np.array([site[time] for site in Prot_time_df[T][1:] if np.ndim(site) > 0]) for time in range(9)]) for T in Prot_time_df.columns]))

d1 = datetime.today()
print(d1.strftime('%d-%m-%Y'))

def MCMC_mu_sin_new(alpha,beta,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs):
    res = pt.zeros((0,), dtype='float64')
    for R in mRNA_df.columns:
        alpha_start,alpha_end = alpha_pos_dict[R]
        a = alpha[alpha_start:alpha_end]
        unit_shape = (len(Target_TFs[R]),9)
        unit = pt.zeros(unit_shape)
        for T in Target_TFs[R]:
            beta_start,beta_end = beta_pos_dict[T]
            if beta_end-beta_start > 1:
                dex = Target_TFs[R].index(T)
                P = pt.sum(beta[beta_start+1:beta_end]*pt.as_tensor(the_prot_dict[T]),axis=1)
                unit = pt.subtensor.set_subtensor(unit[dex],pt.as_tensor(a[dex]* pt.as_tensor(Prot_time_df[T][0]) * (beta[beta_start] + P)))
            else:
                dex = Target_TFs[R].index(T)
                unit = pt.subtensor.set_subtensor(unit[dex],pt.as_tensor(a[dex] * pt.as_tensor(Prot_time_df[T][0]) * beta[beta_start]))
        unit = pt.sum(unit,axis=0)
        res = pt.concatenate([res,unit],axis=0)
    return res


def MCMC_con(alpha,beta,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs):
    unit = pt.zeros((len(alpha_pos_dict)+len(beta_pos_dict),))
    ones = pt.ones((len(alpha_pos_dict)+len(beta_pos_dict),))
    dex = 0
    for k in alpha_pos_dict:
        # ------------------------------------------------------
        unit = pt.subtensor.set_subtensor(unit[dex],pt.sum(alpha[alpha_pos_dict[k][0]:alpha_pos_dict[k][1]]))
    for l in beta_pos_dict:
        #  ------------------------------------------------------
        unit = pt.subtensor.set_subtensor(unit[dex],pt.sum(beta[beta_pos_dict[l][0]:beta_pos_dict[l][1]]))
    res = ones - unit
    return res


if __name__ == '__main__':
    Y_obs = pt.as_tensor(np.array([(mRNA_df[j][t]) for j in mRNA_df for t in range(9)]))
    print(Y_obs)

    last_a = list(alpha_pos_dict.items())[-1][-1][-1]
    print(last_a)
    last_b = list(beta_pos_dict.items())[-1][-1][-1]
    print(last_b)
    
    with pm.Model() as RNA_Model:
        # Priors for unknown model parameters
        alpha = pm.Uniform("alpha", lower=0, upper=1, shape=(last_a,))
        beta = pm.Uniform("beta", lower=-1, upper=1, shape=(last_b,))

        #mu = pt.concatenate([MCMC_mu_sin_new(alpha,beta,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs),MCMC_con(alpha,beta,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs)],axis=0)
        mu = MCMC_mu_sin_new(alpha,beta,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs)
        
        con = MCMC_con(alpha,beta,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs)
        
        c = pt.eq(con, pt.zeros_like(con))
        p = pm.Potential("con",pm.math.log(pm.math.switch(c,1,0.5)))
        
        sigma = Y_obs - mu
        
        # Likelihood of observations
        Y = pm.Normal("Y", mu=mu, sigma=sigma, observed=Y_obs)
        
        # draw 1000 posterior samples
        idata = pm.sample(nuts={'target_accept':0.999,'max_treedepth':30}, draws=1000, cores=4)
    
    d2 = datetime.today()
    dif = d1 - d2
    Time = dif.seconds + (dif.days * 3600 * 24)
    
    az.summary(idata)
    name = 'idata_MCMC_single_HitG_50_scon_{d1}_{Time}'.format(Time=Time,d1=d1.strftime('%Y-%m-%d')))