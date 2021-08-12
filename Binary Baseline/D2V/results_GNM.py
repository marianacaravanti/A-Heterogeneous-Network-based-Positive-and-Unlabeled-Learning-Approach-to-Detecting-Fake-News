import sys
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import urllib.request


class resultsGNM():
    def __init__(self):

        self.exp_id = sys.argv[1]
        self.pulp_id = sys.argv[2]
        self.prop_id = sys.argv[3]
        self.file_name = sys.argv[4]
        dir_bow_op = sys.argv[5]

        exp_metadata = pd.read_csv('params.metadata', sep='\t', index_col=0)
        self.exp_metadata = exp_metadata.loc[self.exp_id]

        pulp_metadata = pd.read_csv(dir_bow_op+'params_pulp.metadata', sep='\t', index_col=0)
        self.pulp_metadata = pulp_metadata.loc[self.pulp_id]

        prop_metadata =  pd.read_csv(dir_bow_op+'params_labelprop.metadata', sep='\t', index_col=0)
        self.prop_metadata = prop_metadata.loc[self.prop_id]
        dir_labels = dir_bow_op.split('/')[:2]
        dir_labels = '/'.join(dir_labels)
        
        self.labels = np.load('dataset/labels.npy')
        self.indexes = np.load('dataset/indexes.npy')

        self.folds = self.pulp_metadata.fold.split(',')
        self.train, self.test = self.calc_train()
            
        self.calc()


    def calc_train(self):
        train = []
        for i in self.folds:
            with open('folds/fold'+i) as f:
                for row in f:
                    train.append(row.replace('\n',''))

        test = []
        with open(self.file_name) as f:
            for row in f:
                name, class_vector = row.split('\t')
                id, layer = name.split(':')
                if(layer == 'news' and id not in train):
                    test.append(row)
        return train, test


    def calc(self):

        tp = fp = tn = fn = 0

        results = []
        labels = []
        

        for f in self.test:
            name, class_vector = f.split('\t')
            name, layer = name.split(':')
            if(layer=='news'):
                index_position = np.where(self.indexes == name)
                label = int(self.labels[index_position[0]])

                x, y = class_vector.split(',')
                x, y = float(x), float(y)
              
        
            if(x > y and label == 1):
                results.append(1)
                labels.append(1)

            elif(x > y and label == -1):
                results.append(1)
                labels.append(-1)

            elif(y > x and label == -1):
                results.append(-1)
                labels.append(-1)

            elif(y > x and label == 1):
                results.append(-1)
                labels.append(1)
          
        matriz_confusao = pd.DataFrame(0, columns={'pred_pos', 'pred_neg'},index={'classe_pos', 'classe_neg'})
        for l in range(len(results)):
            rotulo_original =  'classe_pos' if labels[l] == 1 else 'classe_neg'
            predito = 'pred_pos' if results[l] == 1 else 'pred_neg'
            matriz_confusao[predito][rotulo_original] += 1
                  
          

        
         
        f = open('results/avaliados.csv', 'a')

        id = self.prop_id
        dataset = self.exp_metadata.dataset
        representation = self.exp_metadata.representation_model
        stopwords = self.exp_metadata.stopwords
        language =  self.exp_metadata.language
        option = str(self.exp_metadata.option)
        ngram = str(self.exp_metadata.ngram_range)
        min_df = str(self.exp_metadata.min_df)
        norm = self.exp_metadata.norm
        arg_min_weigth = str(self.exp_metadata.arg_min_weigth)
       
        fold = str(self.pulp_metadata.fold)
        #a = str(self.pulp_metadata.a)
        #m = str(self.pulp_metadata.m)
        #l = str(self.pulp_metadata.l)

        network_relation = str(self.prop_metadata.network_relation)
        mi = str(self.prop_metadata.mi)
        max_iter = str(self.prop_metadata.max_iter)
        limiar_conv = str(self.prop_metadata.limiar_conv)
        weight_relations = str(self.prop_metadata.weight_relations)
        k = str(self.prop_metadata.k)
        
        matriz_confusao.to_csv('results/confusion/confusion_gnm_{}_{}.csv'.format(self.prop_id, str(mi)), sep='\t')
        positive_total = matriz_confusao['pred_pos'].sum() #total de noticias classificadas como positivo
        true_positive_news = matriz_confusao['pred_pos'].loc['classe_pos'].sum()

        list_params_exp = "\t".join([id, dataset, representation, stopwords, language, option, ngram, min_df, norm, arg_min_weigth])
        list_params_pulp = "\t".join([fold, k])
        list_params_prop = "\t".join([network_relation, mi, max_iter, limiar_conv, weight_relations])

        TP = matriz_confusao['pred_pos'].loc['classe_pos'].sum()
        FP = matriz_confusao['pred_pos'].loc['classe_neg'].sum()
        TN = matriz_confusao['pred_neg'].loc['classe_neg'].sum()
        FN = matriz_confusao['pred_neg'].loc['classe_pos'].sum()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2*precision*recall)/(precision+recall)
      
        f.write("\t".join(['gnm', list_params_exp, list_params_pulp, list_params_prop, 
            str(f1_score(labels, results, average='macro')), str(f1_score(labels, results, average='micro')), 
            str(accuracy_score(labels, results)), str(precision_score(labels, results, average='macro')), 
            str(precision_score(labels, results, average='micro')), str(recall_score(labels, results, average='macro')), 
            str(recall_score(labels, results, average='micro')), str(true_positive_news/positive_total), 
            str(precision), str(recall), str(f1)])+'\n')
        
        f.close()
        #url_template = "https://docs.google.com/forms/d/e/1FAIpQLSfQmlzCtMv1LuYL4UE1dTorP_ryFXuKqfE5YrSoW1tKHpEPrA/formResponse?usp=pp_url&entry.696927733={}&entry.344743182={}&entry.1064901768={}&entry.1516025323={}&entry.214559079={}&entry.691617578={}&entry.1273304593={}&entry.1773537280={}&entry.1115242368={}&entry.486506371={}&entry.952006801={}&entry.367300430={}&entry.773657930={}&entry.593706304={}&entry.1156216315={}&entry.249665008={}&entry.912046237={}&entry.714209332={}&entry.376489231={}&entry.1739256340={}&entry.1526187273={}&entry.1491252807={}&entry.1081879557={}&entry.51023073={}&entry.1757061022={}&entry.6961820={}&entry.674034942={}&entry.1470110518={}&entry.1348144936={}&entry.1034163807={}&entry.243179505={}&url_submit=Submit"
        #url_submit = url_template.format("gnm", id, urllib.parse.quote_plus(dataset), urllib.parse.quote_plus(representation), 
        #    urllib.parse.quote_plus(stopwords), urllib.parse.quote_plus(language), option, ngram, min_df, urllib.parse.quote_plus(norm),
        #    arg_min_weigth, fold, k, urllib.parse.quote_plus(network_relation), mi, max_iter, limiar_conv, urllib.parse.quote_plus(str(weight_relations)),
        #    f1_score(labels, results, average='macro'), f1_score(labels, results, average='micro'), 
        #    accuracy_score(labels, results), precision_score(labels, results, average='macro'), 
        #    precision_score(labels, results, average='micro'), recall_score(labels, results, average='macro'), 
        #    recall_score(labels, results, average='micro'), true_positive_news/positive_total, 
        #    precision, recall, f1) 
        
        
        #contents = urllib.request.urlopen(url_submit).read()
        
        print('Macro f1:', f1_score(labels, results, average='macro'))
        print('Micro f1:', f1_score(labels, results, average='micro'))
        print('accuracy:', accuracy_score(labels, results))
        print('Macro Precision:', precision_score(labels, results, average='macro'))
        print('Micro Precision:', precision_score(labels, results, average='micro'))
        print('Macro Recall:', recall_score(labels, results, average='macro'))
        print('Micro Recall:', recall_score(labels, results, average='micro'))
        print('True Positive:', true_positive_news/positive_total)
        print('Interest-class precision: ', precision)
        print('Interest-class recall:', recall)
        print('Interest-class f1:', f1)
        

gnm = resultsGNM()
