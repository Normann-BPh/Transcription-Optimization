'''
Purpose:    construct dataframes for time series of mRNA and TF (including phosphosites);
            construct dictionaries of TF-mRNAs (source x targets) and mRNA-TFs (target x sources) pairs;
            construct dictionaries for parameter indices for equation application
            (not necessary: build network with networkx and pyvis; default: commeted out)
'''

import json # save and read files in json format (human readable)
import numpy as np
import pandas as pd
import networkx as nx # constructing simple node-edge network of TF-mRNA connections
from pyvis import network as net # used to change networkx-network to a html representation for handeling bigger networks

df_Coll = pd.read_csv('../../tabels/CollecTRI.csv') # CollecTRI table from Omnipath in csv format; for more information see https://github.com/saezlab/CollecTRI
'''
+--------+-----------------------+--------+------------+-----------+------------+---------------+---------------+--------------------+-----------------------+---------------------+-----------------------------------+------------------------------------------+----------------+-------------+------------+
|index   |source                 |target  |source_sym  |target_sym |is_directed |is_stimulation |is_inhibition  |consensus_direction |consensus_stimulation  |consensus_inhibition |sources                            |references                                |curation_effort |n_references |n_resources |
+--------+-----------------------+--------+------------+-----------+------------+---------------+---------------+--------------------+-----------------------+---------------------+-----------------------------------+------------------------------------------+----------------+-------------+------------+
| 0      | P01106                | O14746 | MYC        | TERT      | 1          | 1             | 0             | 1                  | 1                     | 0                   | CollecTRI;DoRothEA-A_CollecTRI;...| CollecTRI:10022128;CollecTRI:10491298;...| 82             | 81          | 1          |
+--------+-----------------------+--------+------------+-----------+------------+---------------+---------------+--------------------+-----------------------+---------------------+-----------------------------------+------------------------------------------+----------------+-------------+------------+
| 1      | P17947                | P02818 | SPI1       | BGLAP     | 1          | 1             | 0             | 1                  | 1                     | 0                   | CollecTRI;ExTRI_CollecTRI         | CollecTRI:10022617;ExTRI:                |  3             |  2          | 1          |
+--------+-----------------------+--------+------------+-----------+------------+---------------+---------------+--------------------+-----------------------+---------------------+-----------------------------------+------------------------------------------+----------------+-------------+------------+
| 2      | COMPLEX:P15407_P17275 | P05412 | FOSL1_JUNB | JUN       | 1          | 1             | 0             | 1                  | 1                     | 0                   | CollecTRI;ExTRI_CollecTRI;...     | CollecTRI:10022869;CollecTRI:10037172;   | 53             | 52          | 1          |
+--------+-----------------------+--------+------------+-----------+------------+---------------+---------------+--------------------+-----------------------+---------------------+-----------------------------------+------------------------------------------+----------------+-------------+------------+
'''
df_Gaus = pd.read_pickle('../pickle/MS_Gaus_w_Symbols.pkl') # MS_Gaussian DataFrame with GeneNames
'''
+------+-------+------------+-----------+----------+------+-----------+-------------+------------+---------------+-----+-----------------+-----------------+----------+-------------+-------+
|index |GeneID |mean        |std        |unit_time |time  |experiment |predict_mean |predict_std |type           |site |ENSP             |ENSG             |zscore    |pvalue       |Symbols|
+------+-------+------------+-----------+----------+------+-----------+-------------+------------+---------------+-----+-----------------+-----------------+----------+-------------+-------+
| 1    | 79026 | -0.129678  | 0.0304058 | 0        | 0    | t0hrs     | -0.128355   | 0.0570042  | concentration | nan | ENSP00000367263 | ENSG00000124942 | -4.26491 | 1.9998e-05  | AHNAK |
+------+-------+------------+-----------+----------+------+-----------+-------------+------------+---------------+-----+-----------------+-----------------+----------+-------------+-------+
| 2    | 79026 | -0.143839  | 0.0304105 | 1        | 0.5  | t30sec    | -0.149239   | 0.0557468  | concentration | nan | ENSP00000367263 | ENSG00000124942 | -4.72991 | 2.24622e-06 | AHNAK |
+------+-------+------------+-----------+----------+------+-----------+-------------+------------+---------------+-----+-----------------+-----------------+----------+-------------+-------+
| 3    | 79026 | -0.0552784 | 0.0295396 | 2        | 0.75 | t45sec    | -0.0445536  | 0.0539386  | concentration | nan | ENSP00000367263 | ENSG00000124942 | -1.87133 | 0.0612991   | AHNAK |
+------+-------+------------+-----------+----------+------+-----------+-------------+------------+---------------+-----+-----------------+-----------------+----------+-------------+-------+
'''
df_TS = pd.read_pickle('../pickle/time_series_sym_corr.pkl') # RoutLimma table with GeneNames
'''
+------+-------+-------+------------+-------------+------------+------------+-----------+----------+----------+-----------+-----------+---------------+---------+----------+----------+----------+---------+
|index |GeneID |Length |Min4vsCtrl  |Min8vsCtrl   |Min15vsCtrl |Min30vsCtrl |Hr1vsCtrl  |Hr2vsCtrl |Hr4vsCtrl |Hr8vsCtrl  |Hr16vsCtrl |CtrlHr16vsCtrl |AveExpr  |F         |P.Value   |adj.P.Val |Symbols  |
+------+-------+-------+------------+-------------+------------+------------+-----------+----------+----------+-----------+-----------+---------------+---------+----------+----------+----------+---------+
| 1    |  1647 | 1380  | -0.166817  | -0.00172114 | -0.211775  |  0.475043  | 1.0326    | 1.44752  | 0.583815 | -0.166092 | -0.208262 |  0.430032     | 5.68024 | 108.14   | 1.07e-19 | 1.45e-15 | GADD45A |
+------+-------+-------+------------+-------------+------------+------------+-----------+----------+----------+-----------+-----------+---------------+---------+----------+----------+----------+---------+
| 2    |  6421 | 5832  | -0.0091046 | -0.0920683  | -0.0421459 |  0.0545377 | 0.0656747 | 0.140995 | 0.853116 |  1.03339  |  0.415463 | -0.0406995    | 8.48035 |  77.3507 | 9.85e-18 | 5.22e-14 | SFPQ    |
+------+-------+-------+------------+-------------+------------+------------+-----------+----------+----------+-----------+-----------+---------------+---------+----------+----------+----------+---------+
| 3    | 51129 | 1959  | -0.0160023 | -0.215244   | -0.211535  | -0.251449  | 0.0849374 | 2.12595  | 3.51188  |  1.68208  |  0.45741  | -0.829316     | 2.2728  |  76.4539 | 1.15e-17 | 5.22e-14 | ANGPTL4 |
+------+-------+-------+------------+-------------+------------+------------+-----------+----------+----------+-----------+-----------+---------------+---------+----------+----------+----------+---------+
'''
df_HitG = pd.read_excel('../../tabels/HitGenes.xlsx') # unmodified; collection of genes and their cells respective velocity; used to find interesting, potentially metastasis inducing, genes 
'''
+------+--------+----------+-------------+-------+-------+-----------+--------------+-------+------+-------------+------------------+----+-------+--------------+
|index |gene    |F         |PR(>F)       |column |id     |mean_var   |mean_velocity |n_gene |n_neg |neg_mean_var |neg_mean_velocity |row |set    |DeltaVelocity |
+------+--------+----------+-------------+-------+-------+-----------+--------------+-------+------+-------------+------------------+----+-------+--------------+
| 0    | ISX    | 130.133  | 3.83e-08    |  7    | 91464 | 0.0228607 | 0.529252     | 3     | 12   | 0.678421    | 12.708           | 4  | DG_24 | -12.1787     |
+------+--------+----------+-------------+-------+-------+-----------+--------------+-------+------+-------------+------------------+----+-------+--------------+
| 1    | ZBTB45 | 115.511  | 7.75e-08    |  1    | 84878 | 0.192954  | 0.854429     | 3     | 12   | 0.678421    | 12.708           | 1  | DG_24 | -11.8536     |
+------+--------+----------+-------------+-------+-------+-----------+--------------+-------+------+-------------+------------------+----+-------+--------------+
| 2    | MMS19  |  23.9139 | 0.000479047 | 20    | 64210 | 3.93e-10  | 0            | 1     | 12   | 0.171521    | 11.7152          | 5  | DG_22 | -11.7152     |
+------+--------+----------+-------------+-------+-------+-----------+--------------+-------+------+-------------+------------------+----+-------+--------------+
'''
df_HitG = df_HitG.sort_values(by=['mean_velocity','PR(>F)'],ascending=[False,True]) #focus on mean_v, should there be equals go for PR

gene_list_HitG = list(df_HitG['gene']) # list of the 'gene'-column of HitG

edge_list = [] # will contain TF-mRNA pairs, i.e. edges of the network
target_list = list(df_Coll['target_genesymbol'])
target_df_excerpt = df_Coll['target_genesymbol']
mRNA_symbols_list = list(df_TS['Symbols'])
source_list = list(df_Coll['source_genesymbol'])
source_df_excerpt = df_Coll['source_genesymbol']
TF_symbols_list = list(df_Gaus['Symbols'])
not_found = [] # will contain TFs or mRNA not found in CollecTRI and Gaus/RoutLimma
i = 0

# checks if the target or source is in CollecTRI; adds pairs to the edge_list
for i in range(len(gene_list_HitG)):
    if gene_list_HitG[i] in source_list:
        indices_to_check = list(source_df_excerpt[source_df_excerpt == gene_list_HitG[i]].index)
    elif gene_list_HitG[i] in target_list:
        indices_to_check = list(target_df_excerpt[target_df_excerpt == gene_list_HitG[i]].index)
    try:
        for j in indices_to_check:
            if source_df_excerpt.loc[j] in TF_symbols_list and target_df_excerpt.loc[j] in mRNA_symbols_list:
                edge_list.append((source_df_excerpt.loc[j], target_df_excerpt.loc[j]))
    except:
        print('nan',gene_list_HitG[i])
        not_found.append(gene_list_HitG[i])
        continue

with open('not_found_HitG','w') as f:
    json.dump(not_found,f)

print('Length of edge list: ',len(edge_list))


# preDictionaries #

Target_TFs = dict()
TF_Targets = dict()
for i in edge_list:
    Source, Target = i[0], i[1]
    if Target in Target_TFs:
        Target_TFs[Target].add(Source)
    else:
        Target_TFs[Target] = {Source}
    if Source in TF_Targets:
        TF_Targets[Source].add(Target)
    else:
        TF_Targets[Source] = {Target}
for i in Target_TFs:
    Target_TFs[i] = list(Target_TFs[i])
for i in TF_Targets:
    TF_Targets[i] = list(TF_Targets[i])

'''
## TF_Targets ##
TF_Targets = dict()
for i in edge_list:
    Source, Target = i[0], i[1]
    if Source in TF_Targets:
        TF_Targets[Source].add(Target)
    else:
        TF_Targets[Source] = {Target}
for i in TF_Targets:
    TF_Targets[i] = list(TF_Targets[i])
'''

# Protein DF #

Prot_df = pd.DataFrame()
nothing = []
non_log_fold_change = []
names = []
for i in Target_TFs:
    for j in Target_TFs[i]:
        if str(j) in names: # skip if already in df
            continue
        if df_Gaus[df_Gaus['Symbols'] == j].index.size < 1: # skip if the symbol can't be found
            nothing.append(j)
            continue
        elif df_Gaus[df_Gaus['Symbols'] == j].index.size >= 1: # find symbol
            names.append(str(j)) # add to names
            temp_list = []
            start_index = df_Gaus[df_Gaus['Symbols'] == j].index.values[0] - 1 
            index = start_index
            while df_Gaus.iloc[index, 14] == j: # find the end
                end_index = index
                index += 1
            for k in range(start_index, end_index, 15):
                k += 5
                l = 0
                s = []
                if str(df_Gaus.iloc[k]['site'])[0] in ['S','T','Y'] or str(df_Gaus.iloc[k]['site']) == 'nan':
                    while l < 9:
                        s.append(2**float(df_Gaus.iloc[k+l, 6]))
                        l += 1
                    temp_list.append(s)
                else:
                    continue
            non_log_fold_change_series = pd.Series(temp_list)
            non_log_fold_change.append(non_log_fold_change_series)
Prot_df = pd.concat(non_log_fold_change, axis = 1)
Prot_df.columns = names
Prot_time_df = Prot_df.reindex(sorted(Prot_df.columns), axis=1)

## not optimizable ##
non_opti = []
for i in Target_TFs:
    if len(Target_TFs[i]) < 2:
        j = Target_TFs[i][0]
        if len([k for k in Prot_df[j] if str(k) != 'nan']) < 2:
            non_opti.append(i)
with open('./dicts/non_optimizables_HitG.json', 'w') as f:
    json.dump(non_opti,f)
print('# not optimizable: ',len(non_opti))


# Update Dictionaries #

## Target_TFs ##
for i in non_opti:
    if i in Target_TFs:
        del Target_TFs[i]

## TF_Targets ##
for i in TF_Targets:
    for j in non_opti:
        if j in TF_Targets[i]:
            TF_Targets[i].remove(j)


# mRNA DF #

mRNA_df = pd.DataFrame()
nothing = []
non_log_fold_change = []
names = []
for i in Target_TFs:
        if df_TS[df_TS['Symbols'] == i].index.size < 1:
                nothing.append(i)
                continue
        elif df_TS[df_TS['Symbols'] == i].index.size >= 1:
                names.append(str(i))
                index = df_TS[df_TS['Symbols'] == i].index.values[0] - 1
                non_log_fold_change_series = pd.Series([(2**float(i)) for i in list(df_TS.iloc[index])[2:11]])
                non_log_fold_change.append(non_log_fold_change_series)
mRNA_df = pd.concat(non_log_fold_change, axis = 1)
mRNA_df.columns = names
mRNA_df = mRNA_df.reindex(sorted(mRNA_df.columns), axis=1)


# Update Dictionaries #

## Target_TFs ##
for i in nothing:
    if i in Target_TFs:
        del Target_TFs[i]
with open('./dicts/Target_TFs_HitG.json', 'w') as f:
    json.dump(Target_TFs,f)
print('Available mRNAs: ',len(Target_TFs))

## TF_Targets ##
for i in TF_Targets:
    for j in nothing:
        if j in TF_Targets[i]:
            TF_Targets[i].remove(j)
with open('./dicts/TF_Targets_HitG.json', 'w') as f:
    json.dump(TF_Targets,f)
print('Available TFs: ',len(TF_Targets))


# threshold definition and saving mRNA_df #

threshold = int(input('Enter amount of wanted mRNAs, values must be between 0 and {}: '.format(len(Target_TFs))))
to_keep = list(Target_TFs.keys())[:threshold] # chose first 50 in speed for thesis equivalent results
mRNA_df = mRNA_df[to_keep]
mRNA_df.to_pickle('../pickle/mRNA_collectri_HitG_{}_df.pkl'.format(threshold)) # saved double for safety
mRNA_df.to_pickle('./pickle/mRNA_collectri_HitG_{}_df.pkl'.format(threshold))


# update Prot_df #

s = list()
for mRNA_symbols_list in mRNA_df.columns:
    for TF in Target_TFs[mRNA_symbols_list]:
        s.append(TF)
s_ = set(s)
Prot_time_df = Prot_time_df[list(s_)]
Prot_time_df = Prot_time_df.reindex(sorted(Prot_time_df.columns), axis=1)
Prot_time_df.to_pickle('../pickle/Prot_time_collectri_HitG_{}_df.pkl'.format(threshold)) # saved double for safety
Prot_time_df.to_pickle('./pickle/Prot_time_collectri_HitG_{}_df.pkl'.format(threshold))



'''
# was excluded since not necessary to run everytime when preprocessing; can be enabled by uncommenting
# if not working properly, check if the templates are proper and in use; or consult the docummentation of pyvis (https://pyvis.readthedocs.io/en/latest/index.html)
# Edge list for the chosen threshold
new_edge_list = []
for i in edge_list:
    if i[1] in to_keep:
        new_edge_list.append(i)

TF_symbols_list = nx.MultiDiGraph()
TF_symbols_list.add_edges_from(new_edge_list)
TF_symbols_list.edges()

nt = net.Network(height="1080px", width="177%",notebook = False, 
                bgcolor="#222222", font_color="violet", filter_menu=True, cdn_resources='in_line')
for i in non_opti:
    nt.add_node(i, label=i, shape='box', color='orange')
for i in nothing:
    nt.add_node(i, label=i, shape='box', color='red')
for i in to_keep:
    nt.add_node(i, label=i, shape='circle', color='green')
nt.from_nx(TF_symbols_list)
nt.force_atlas_2based()
nt.toggle_physics(True)
template_dir = 'C:/Users/lvci/seadrive_root/Julius N/My Libraries/Bachelor/code/templates'
nt.set_template_dir(template_dir, template_file='no_move_template.html')
nt.show_buttons()
link_html = './plots/Network_TFT_interactions_fa2b_loops_colour_HitG_{}.html'.format(threshold)
nt.save_graph('./plots/Network_TFT_interactions_fa2b_loops_colour_HitG.html')
'''



'''
The following lines construct dictionaries containing the mRNA and each of the parameter-indices for the respective TFs (alpha-dictionaries)
and the dictionaries for the TFs and each of the parameter-indices for their respective phosphosites (beta-dictionaries)
Additionally to the difference between the RMSD and single equation, the methods use different equation structures and thus have different indices
for the parameters (SLSQP and DE use the same equations). As to not confuse myself they are seperated into these chunks.
'''
# number of variables #

## alpha ##
c_a = 0
for i in to_keep:
    c_a += len(Target_TFs[i])
##beta:
c_b = 0
for i in Prot_time_df:
    c_b += len([j for j in Prot_time_df[i] if str(j) != 'nan'])
##equations:
c_e = len(to_keep)


# SLSQP & DE #

## rmsd ##

### alpha/beta - dictionary ###
fp_start = 0
fm_start = c_e
alpha_start = fm_start + c_e
beta_start = alpha_start + c_a
error_pos_dict = {}
beta_pos_dict = {}
alpha_pos_dict = {}
for i in mRNA_df.columns:
    error_pos_dict[i] = [fp_start,fm_start]
    fp_start,fm_start = fp_start+1,fm_start+1
    ####alpha:
    alpha_end = alpha_start + len(Target_TFs[i])
    alpha_pos_dict[i] = [alpha_start,alpha_end]
    alpha_start = alpha_end
    ####beta:
    for j in Target_TFs[i]:
        if j not in beta_pos_dict:
            beta_pos_dict[j] = [beta_start,beta_start + len([k for k in Prot_time_df[j] if str(k) != 'nan'])]
            beta_start = beta_start + len([k for k in Prot_time_df[j] if str(k) != 'nan'])

link_dict = './dicts/SQP/SQP_HitG_{}_rmsd_'.format(threshold)
with open(link_dict+'alpha_dict.json','w') as f:
    json.dump(alpha_pos_dict,f)
with open(link_dict+'beta_dict.json','w') as f:
    json.dump(beta_pos_dict,f)
with open(link_dict+'error_dict.json','w') as f:
    json.dump(error_pos_dict,f)

## single ##

### alpha/beta - dictionary ###
fp_start = 0
fm_start = c_e*9
alpha_start = fm_start + c_e*9
beta_start = alpha_start + c_a
error_pos_dict = {}
beta_pos_dict = {}
alpha_pos_dict = {}
for i in mRNA_df.columns:
    error_pos_dict[i] = [fp_start,fm_start]
    fp_start,fm_start = fp_start+9,fm_start+9
    ####alpha:
    alpha_end = alpha_start + len(Target_TFs[i])
    alpha_pos_dict[i] = [alpha_start,alpha_end]
    alpha_start = alpha_end
    ####beta:
    for j in Target_TFs[i]:
        if j not in beta_pos_dict:
            beta_pos_dict[j] = [beta_start,beta_start + len([k for k in Prot_time_df[j] if str(k) != 'nan'])]
            beta_start = beta_start + len([k for k in Prot_time_df[j] if str(k) != 'nan'])

link_dict = './dicts/SQP/SQP_HitG_{}_single_'.format(threshold)
with open(link_dict+'alpha_dict.json','w') as f:
    json.dump(alpha_pos_dict,f)
with open(link_dict+'beta_dict.json','w') as f:
    json.dump(beta_pos_dict,f)
with open(link_dict+'error_dict.json','w') as f:
    json.dump(error_pos_dict,f)


# NSGA2 #

## rmsd & single ##

### alpha/beta - dictionary ###
alpha_start = 0
beta_start = alpha_start + c_a
beta_pos_dict = {}
alpha_pos_dict = {}
for i in mRNA_df.columns:
    ####alpha:
    alpha_end = alpha_start + len(Target_TFs[i])
    alpha_pos_dict[i] = [alpha_start,alpha_end]
    alpha_start = alpha_end
    ####beta:
    for j in Target_TFs[i]:
        if j not in beta_pos_dict:
            beta_pos_dict[j] = [beta_start,beta_start + len([k for k in Prot_time_df[j] if str(k) != 'nan'])]
            beta_start = beta_start + len([k for k in Prot_time_df[j] if str(k) != 'nan'])
link_dict = './dicts/GA/GA_HitG_{}_'.format(threshold)
with open(link_dict+'alpha_dict.json','w') as f:
    json.dump(alpha_pos_dict,f)
with open(link_dict+'beta_dict.json','w') as f:
    json.dump(beta_pos_dict,f)


# MCMC #

## single ##

### alpha/beta - dictionary ###
alpha_start = 0
beta_start = 0
beta_pos_dict = {}
alpha_pos_dict = {}
for i in mRNA_df.columns:
    ####alpha:
    alpha_end = alpha_start + len(Target_TFs[i])
    alpha_pos_dict[i] = [alpha_start,alpha_end]
    alpha_start = alpha_end
    ####beta:
    for j in Target_TFs[i]:
        if j not in beta_pos_dict:
            beta_pos_dict[j] = [beta_start,beta_start + len([k for k in Prot_time_df[j] if str(k) != 'nan'])]
            beta_start = beta_start + len([k for k in Prot_time_df[j] if str(k) != 'nan'])

link_dict = './dicts/MCMC/MCMC_HitG_{}_'.format(threshold)
with open(link_dict+'alpha_dict.json','w') as f:
    json.dump(alpha_pos_dict,f)
with open(link_dict+'beta_dict.json','w') as f:
    json.dump(beta_pos_dict,f)