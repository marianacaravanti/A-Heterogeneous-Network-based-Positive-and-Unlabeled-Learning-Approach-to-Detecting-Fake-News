from util.bibliotecas import *
from util.functions import *
from sklearn import preprocessing


class MatricesCalc():
	def __init__(self, exp_id, exp_metadata, net_metadata):

		option = exp_metadata.option
		self.language = exp_metadata.language
		input_path = sys.argv[2].format(option,exp_id)
		input_files = input_path.split('/')[:2]
		input_files = "/".join(input_files)
		
		self.net_metadata = net_metadata
		self.bow_file = input_path+'tf_idf_vect.csv'
		self.adj_matrix_file = input_path+'representation_input/adj_matrix.csv'
		self.graphml_output_file = input_path+'representation_input/{}/knn_fakenews.graphml'
		self.doc_term_relations_output = input_path+'representation_input/{}/' 	
		self.output_matriz_w = input_path+'representation_input/{}/{}/w_matrix.csv'
		self.data_features = pd.read_csv("dataset/news_features.csv", sep='\t', header=0, index_col=0)		
		
		arg_k = sys.argv[3].split(',')
		arg_a = sys.argv[4].split(',')

		self.arg_k = [int(x) for x in arg_k]
		self.arg_a = [float(x) for x in arg_a]
		self.arg_min_weigth = float(exp_metadata.arg_min_weigth)
		self.indexes_file = 'dataset/indexes.npy'
		self.labels_file = 'dataset/labels.npy'
		
		self.indexes = np.load(self.indexes_file)
		self.labels = np.load(self.labels_file)

		self.tfidf_vect_df = pd.read_csv(self.bow_file, sep=',', index_col=0, header=0)

		self.adjacence_matrix = self.calc_MatrizAdj()
		self.ident_matrix = self.calc_Indent()
		self.A = None
		self.G = None
		self.nodes_map = {}
		self.G_features = None
		self.calc_knn_matrices()	
		
	def calc_Indent(self):
		print("Calculando matriz identidade...", flush=True)
		ident_matrix = np.identity(len(self.adjacence_matrix), dtype = float)
		ident_matrix = pd.DataFrame(ident_matrix, columns=self.indexes, index=self.indexes)
		return ident_matrix


	def calc_MatrizAdj(self):
		adj_matrix = []
		for id in self.indexes:
			adj_matrix.append(self.tfidf_vect_df.loc[id].tolist())

		adj_matrix = pd.DataFrame(adj_matrix, index=self.indexes)

		time_init = time()
		Y = cdist(adj_matrix, adj_matrix, metric=cosine)
		adjacence_matrix = pd.DataFrame(Y, columns=self.indexes, index=self.indexes)
		adjacence_matrix.to_csv(path_or_buf=self.adj_matrix_file)
		return adjacence_matrix

	

	def calc_knn_matrices(self):
		
		self.calc_features_net()
		
		print("Calculating k-nn matrices...", flush=True)
		for valor_k in self.arg_k:

			print("Calculando {}-NN...".format(valor_k), flush=True)
			self.A, self.G = self.PU_LP_knn(valor_k)

			self.create_networks(valor_k)

			for alfa in self.arg_a:
				self.calc_W(valor_k, alfa)
			
			#------------------------------------------------
	
	def calc_W(self, valor_k, alfa):
		matrix_w_output_name = self.output_matriz_w.format(valor_k, alfa)
		
		print("\tCalculando ??ndice de Katz aplha={}...".format(alfa), flush=True)
		#Calcula W = (I - alfa x A)^-1 - I
		#Medida de centralidade que considera n??mero de vizinhos de um n?? e o qu??o importante esses vizinhos s??o
		A = self.A.mul(alfa)
		I = self.ident_matrix.subtract(A)
		
		I = pd.DataFrame(np.linalg.pinv(I.values.astype(np.float32)), columns=self.indexes, index=self.indexes)
		W = I.subtract(self.ident_matrix)

		W.to_csv(path_or_buf=matrix_w_output_name)

		print('\t\tDesalocar Matriz W', flush=True)
		del W	

	
	def calc_features_net(self):
		self.G_features = nx.DiGraph()
		
		features = self.net_metadata.features[len(self.net_metadata)-1].split(',')
		for i in features: self.nodes_map[i]=[]
		for f in features:
			if f == 'news': continue
			elif f == 'term':
				for term in self.tfidf_vect_df.columns:
					for j in range(len(self.indexes)):
						news = self.indexes[j]
						weight = self.tfidf_vect_df.loc[news,term]
						if weight >= self.arg_min_weigth:
							node_1 = news + ':news'
							node_2 = term + ':term'      
							self.G_features.add_edge(node_1,node_2, weight=weight)
							self.G_features.add_edge(node_2,node_1, weight=weight)		
							self.nodes_map[f].append(node_2)		
					
			else:
				list_features = self.data_features[f]
				norm_features = [j/list_features.max() for j in list_features]
				list_normalized = pd.DataFrame(norm_features, index=self.data_features.index)
				self.nodes_map[f].append(f+':'+f)					
				for j in range(len(self.indexes)):
					news = self.indexes[j]
					weight = list_normalized.loc[news,0]
					if weight >= self.arg_min_weigth: 
						node_1 = news + ':news'
						node_2 = f + ':'+f   
						self.G_features.add_edge(node_1,node_2, weight=weight)
						self.G_features.add_edge(node_2,node_1, weight=weight)	
		
	
	def create_networks(self, valor_k):		
		
		for i in range(len(self.net_metadata)):
			G_het = nx.DiGraph()
			
			
			doc_term_relations_output = self.doc_term_relations_output.format(valor_k)+self.net_metadata.network_name[i]
			fwrite = open(doc_term_relations_output, 'w')
			features = self.net_metadata.features[i].split(',')

		
			for f in features:
				if f == 'news': 
					for x, y in self.G.edges:
						G_het.add_edge(x+':news', y+':news', weight=self.G[x][y]['weight'])
					
				else:
					for node in self.nodes_map[f]:
						for vizinho in self.G_features.neighbors(node):
							G_het.add_edge(node,vizinho,weight=self.G_features[node][vizinho]['weight'])
							G_het.add_edge(vizinho,node,weight=self.G_features[node][vizinho]['weight'])
							

			self.normalize_relations(G_het, features)
			for x, y in G_het.edges:
				fwrite.write("{}\t{}\t{}\n".format(x, y, G_het[x][y]['weight']))
			fwrite.close()


	def normalize_relations(self, G_het, features):
		for f in features:
			for node in self.nodes_map[f]:
				for l in features:
					#print(f,l)
					soma = 0
					for vizinho in G_het.neighbors(node):
						if vizinho.split(':')[1] == l:
							#print('weight:', G_het[node][vizinho]['weight'])
							soma+=G_het[node][vizinho]['weight']
							#print(soma)

					for vizinho in G_het.neighbors(node):
						if vizinho.split(':')[1] == l:
							weight = G_het[node][vizinho]['weight']
							#print('antigo weight:', G_het[node][vizinho]['weight'])
							G_het[node][vizinho]['weight'] = weight/soma
		
							



				
	def PU_LP_knn(self, k):
		
		total_columns = self.adjacence_matrix.shape[0]
		#A ?? uma matriz com base em k-NN, na qual Aij = 1 se j ?? um dos k vizinhos de i, e 0 c.c.
		A = pd.DataFrame(0, columns=self.adjacence_matrix.index, index=self.adjacence_matrix.index)
		#G ?? o grafo gerado a partir da matriz A, cujos v??rtices n??o est??o rotulados
		G = nx.DiGraph()
		

		for index_i, row in self.adjacence_matrix.iterrows():
			self.nodes_map['news'].append(index_i+':news')
			#knn ?? um vetor de k posi????es, inicializadas com valor alto
			#o vetor knn armazena os k ??ndices de vizinhos mais pr??ximos do v??rtice i (menor dist??ncia de cosseno)
			knn = [1000 for temp in range(k)]
			#knn_names armazena os nomes correspondentes aos ??ndices armazenados no vetor knn
			knn_names = ['' for temp in range(k)]
			max_value = 1000
			max_value_id = 0
			#para cada coluna correspondente a linha i (j vizinhos)
			for name_j, value in row.iteritems():
				#se i != j fa??a
				if(index_i != name_j):
					#se a dist??ncia ?? menor que o maior valor armazenado no vetor knn:
					if (value < max_value):
						#adiciono v??rtice j aos vizinhos de i
						knn_names[max_value_id] = name_j
						knn[max_value_id] = value
						max_value_id = np.argmax(knn)
						max_value = knn[max_value_id]
			for j in range(k):
				#Seleciona os 4 vizinhos mais pr??ximos e os adiciono na matriz A
				vizinho = knn_names[j]
				A.loc[index_i][vizinho] = 1
				#Adiciona aresta no grafo entre i e seus 4 vizinhos mais pr??ximos
				#print(index_i, '\t', vizinho)
				G.add_edge(index_i, vizinho, weight=(1-self.adjacence_matrix.loc[index_i][vizinho]))
				G.add_edge(vizinho, index_i, weight=(1-self.adjacence_matrix.loc[index_i][vizinho]))
		
		print("\tSalva informa????es do Grafo")
		nx.write_graphml(G, self.graphml_output_file.format(k))
		return A, G

#Par??metros de entrada
exp_id = sys.argv[1]
exp_metadata = pd.read_csv('params.metadata', sep='\t', index_col=0)
exp_metadata = exp_metadata.loc[exp_id]
net_metadata = pd.read_csv('params_features.metadata', sep='\t', header=0)

mtc = MatricesCalc(exp_id, exp_metadata, net_metadata)

