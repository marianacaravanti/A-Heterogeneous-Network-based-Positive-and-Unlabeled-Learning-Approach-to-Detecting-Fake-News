import numpy as np
import pandas as pd
import networkx as nx
import os, sys


'''def altera_valores(df, column):
	list = []
	for i in range(df.shape[0]):
		x = df[column][i] 
		if x > 1.0:
			x = '0.'+str(int(x))
			x = float(x)
		list.append(x)
	df[column] = list'''

file = 'results/avaliados.csv'

list_index = {0:'algprop', 1:'id', 2:'dataset', 3:'exp_name', 4:'stopwords', 5:'lang', 6:'option', 7:'ngram', 
8:'min_df', 9:'norm', 10:'arg_min_weigth', 11:'fold', 12:'k', 13:'network', 14:'mi', 15:'max_iter', 16:'limiar_conv', 17:'weight_rel', 
18:'macro_f1', 19:'micro_f1', 20:'acuracy', 21:'macro_precision', 22:'micro_precision', 23:'macro_recall', 24:'micro_recall', 25:'tp', 
26:'interest_prec', 27:'interest_recall', 28:'interest_f1'}

df = pd.read_csv(file, sep='\t', index_col=None, header=None, dtype={11: str}).rename(columns=list_index).fillna(0)
df['training_percent'] = df.fold.astype('str').str.len() 

'''altera_valores(df, 'macro_f1')
altera_valores(df, 'micro_f1')
altera_valores(df, 'acuracy')
altera_valores(df, 'macro_precision')
altera_valores(df, 'micro_precision')
altera_valores(df, 'macro_recall')
altera_valores(df, 'micro_recall')
altera_valores(df, 'tp')
altera_valores(df, 'interest_prec')
altera_valores(df, 'interest_recall')
altera_valores(df, 'interest_f1')'''



metric = 'interest_f1'
num_labeles_exes = df['training_percent'].unique()
df_groups = []

num_algs = ['lphn','gnm']

nets = df['network'].unique()

table_results = []
#table_results.append(['network','lphn_10','lphn_20','lphn_30','gnm_10','gnm_20','gnm_30'])

print('>>>', metric)
for net in nets:
	net_result = []
	net_result.append(net)
	for nalg in num_algs:
		for nle in num_labeles_exes:
			df_selected = df[df['network'] == net]
			df_selected = df_selected[df_selected['training_percent'] == nle]
			df_selected = df_selected[df_selected['algprop'] == nalg]
			df_grouped = df_selected.sort_values(by=['dataset', 'exp_name','ngram', 'min_df', 'k','fold'])
			df_grouped = df_grouped.drop(['fold', 'id'], axis=1).groupby(['dataset', 'exp_name','ngram', 'min_df', 'norm', 'k','arg_min_weigth', 'mi', 'max_iter', 'limiar_conv', 'weight_rel', 'arg_min_weigth' ]).mean()
			net_result.append(df_grouped[metric].max())
	table_results.append(net_result)

df_results = pd.DataFrame(table_results, columns=['network','lphn_10','lphn_20','lphn_30','gnm_10','gnm_20','gnm_30'])
print(df_results)

table_results = []

metric = 'macro_f1'
print('>>>', metric)
for net in nets:
	net_result = []
	net_result.append(net)
	for nalg in num_algs:
		for nle in num_labeles_exes:
			df_selected = df[df['network'] == net]
			df_selected = df_selected[df_selected['training_percent'] == nle]
			df_selected = df_selected[df_selected['algprop'] == nalg]
			df_grouped = df_selected.sort_values(by=['dataset', 'exp_name','ngram', 'min_df', 'k','fold'])
			df_grouped = df_grouped.drop(['fold', 'id'], axis=1).groupby(['dataset', 'exp_name','ngram', 'min_df', 'norm', 'k','arg_min_weigth', 'mi', 'max_iter', 'limiar_conv', 'weight_rel', 'arg_min_weigth' ]).mean()
			net_result.append(df_grouped[metric].max())
	table_results.append(net_result)

df_results = pd.DataFrame(table_results, columns=['network','lphn_10','lphn_20','lphn_30','gnm_10','gnm_20','gnm_30'])
print(df_results)
