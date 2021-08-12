import numpy as np
import pandas as pd
import networkx as nx
import sys


class PULP():
	def __init__(self, exp_metadata):
		#graphml_input_file = sys.argv[2] 
		#matrix_adj_input_name = sys.argv[3] 
		self.output_file = sys.argv[3]

		self.train_folds = exp_metadata.fold.split(',')
		#self.m = int(exp_metadata.m)
		#self.lmbda = float(exp_metadata.l)
		self.dataset_file = exp_metadata.dataset

		#self.G = nx.read_graphml(graphml_input_file)
		#self.Adj = pd.read_csv(matrix_adj_input_name, index_col=0)

		self.dataset = None
		self.train = None
		self.test = None
		self.labels = None


	def begin(self):

		print('Lendo arquivo tsv...')
		
		self.dataset = pd.read_csv(self.dataset_file, sep='\t', index_col=0, header=None)

		self.train, self.test, self.labels = self.train_teste()

		self.calc_pulp()


	def calc_pulp(self):
		#P = conjunto de exemplos positivos rotulados
		P = []
		N = []
		L = list(self.train.index.values)
		for i in L:
			label = self.dataset.loc[i][2]
			if label == 1: P.append(i)
			else: N.append(i)

		print('Salvando conjuntos P, N')

		self.save_file(P, N)

		

	def save_file(self, RP, RN):
		f = open(self.output_file, 'w')
		for i in RP: 
			f.write(i + ':news\t1,0\n')

		for i in RN:
			f.write(i + ':news\t0,1\n')
		f.close()



	def train_teste(self):
		train = []
		for i in self.train_folds:
			file = 'folds/fold'+str(i)
			f = open(file, 'r')
			for row in f:
				index  = row.replace('\n','')
				

				train.append(index)
			f.close()

		print("Calculando treino e teste...")
		test = self.dataset.drop(train)
		train = self.dataset.loc[train]

		labels = []
		for index in test.index:
			label = self.dataset.loc[index][2]
			if label == 1:
				labels.append(1)
			else:
				labels.append(-1)
		return train, test, labels


#Parâmetros de entrada
exp_id = sys.argv[1]
metadata = sys.argv[2] 

exp_metadata = pd.read_csv(metadata, sep='\t', index_col=0)
exp_metadata = exp_metadata.loc[exp_id]

pulp = PULP(exp_metadata)
pulp.begin()

