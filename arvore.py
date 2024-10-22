import sys
import math
import pandas as pd
from node import Node
from ID3 import ID3
import random
import numpy as np

def print_tree(node, i, edge):
    if node.is_leaf:
        print("[{}]{}{} {}".format(i, " "*i*4, edge, node.target_value))
    else:
        node_name = node.attribute
        if node.numeric_condition:
            node_name += "(<= " + "{:.2f})".format(node.numeric_condition)
        print("[{}]{}{} '{}'(gain: {:.3f} bits)".format(i, " "*i*4, edge, node_name , node.gain, ))
        for key, val in node.child.items():
            print_tree(val, i+1, str(key)+" ->")

class Arvore:
    def __init__(self, train, target_attr, selection_algorithm, m=None):
        self.root_node = create_tree(train, target_attr, selection_algorithm, m)

    def print(self):
        print_tree(self.root_node,0, "")

    def predict(self, instancia):
        return self.root_node.predict(instancia)



def create_tree(df, target_attr, selection_algorithm, m=None):
    if not m:
        m = (len(df.columns)-1)**(1/2)

    node= Node()
    most_freq_val = df[target_attr].value_counts().idxmax()

    node.set_target_value(most_freq_val)            #guarda valor mais frequente do atributo alvo

#    print("NEW_NODE:")

    for value, sub_df in df.groupby(target_attr):
#        print("{}: {} items".format(value, len(sub_df.index)))
        if len(sub_df.index) == len(df.index):   #tamanho do df dividido eh o mesmo que do original, homogeneo
            node.set_leaf()
            return node

    if df.shape[1] == 1: #nao há novos atributos
        node.set_leaf()
        return node

    attr_list = list(df.columns).copy()
    attr_list.remove(target_attr)

    sampled_attr_list = amostragem_atributos(attr_list, m)


    attr, gain = selection_algorithm(df[sampled_attr_list+[target_attr]].copy(), target_attr)
#    print('choosen attribute: {} gain: {}'.format(attr, gain))

    node.set_attribute(attr)
    node.set_gain(gain)

    if df[attr].dtypes == 'object' or str(df[attr].dtypes) == 'category' :
        condition = attr
    else:
        mean= df[attr].mean()
        node.set_numeric_condition(mean)
        condition = df[attr] <= mean

    for attr_val, df_splited in df.groupby(condition): #divide as instancias pelos valores do atributo selecionado
        if len(df_splited.index) != 0:
            node.child[attr_val] = create_tree(df_splited.drop(columns=attr), target_attr, selection_algorithm, m)
        
    return node

def bootstrap_table(treino):
    n_dados_teste = treino.shape[0]
    return treino.sample(n=n_dados_teste, replace = True)

def out_of_bag_table(df_train, escolhidos):
    #seleciona conjunto de teste
    return df_train[~df_train.index.isin(escolhidos.index)]

def amostragem_atributos(key_list, m):
    n_atri = len(key_list)
    num_samples = int(min(n_atri, m))
    if num_samples == 0:
        num_samples = 1

    new_key_list=[]
    #print("São ",n_atri," atributos")
    escolhidos = random.sample(range(0, n_atri), k=num_samples)#amostragem sem reposição
    for i in range(len(escolhidos)):
        #print(i," escolhido ",escolhidos[i],"esse",key_list[escolhidos[i]],"e esse",type_list[escolhidos[i]])
        new_key_list.append(key_list[escolhidos[i]])

#    print("atributos sorteados", new_key_list)
    return new_key_list

class FlorestaAleatoria:
    def __init__(self, df_train, target_coluna, n_arvores):
        self.floresta=[]

        key_list = list(df_train.columns)
        self.alvo = df_train[key_list[target_coluna]].unique()
#        print("ALVO: ",self.alvo)


        for i in range(n_arvores):
            #seleciona conjuntos de treinamento e teste
            bootstrap = bootstrap_table(df_train.copy())
#            print("BOOTSTRAP:")
#            print(bootstrap)
#            out_of_bag = out_of_bag_table(df_train.copy(), bootstrap)
#            print("OUT OF BAG:")
#            print(out_of_bag)
            #gera a arvore
            arvore = Arvore(bootstrap, bootstrap.columns[target_coluna], ID3)
#            arvore.print()
            self.floresta.append(arvore)

    def predict(self, instancia):
        votacao=[]
        for arvore in self.floresta:
            predicted_value = arvore.predict(instancia)
            votacao.append(predicted_value)
            #print(votacao,"votos")#, divide em ",int(n_arvores/len(alvo)))

        if (len(self.alvo)==2):
            for categoria_alvo in self.alvo:
                if (votacao.count(categoria_alvo) > int(len(self.floresta)/len(self.alvo))):
                    predicted_value=categoria_alvo
                    break
        else:
            mais_votado=self.alvo[-1]
            for categoria_alvo in self.alvo:
                if (votacao.count(categoria_alvo) > votacao.count(mais_votado)):
                    mais_votado=categoria_alvo            
            predicted_value=mais_votado
            
        return predicted_value


# ---------------------------------------------------------------------
dataset = {}
dataset["votos"] = {\
    "data": ('house-votes-84.tsv', '\t'), \
    "types": ('house-votes-84_types.csv', ';')}
dataset["jogo"] =  { \
    "data": ('dadosBenchmark_validacaoAlgoritmoADv2.csv', ';'), \
    "types": ('AttributeType.csv', ';')}
dataset["jogo_original"] =  { \
    "data": ('dadosBenchmark_validacaoAlgoritmoAD.csv', ';'), \
    "types": ('AttributeType_original.csv', ';')}
dataset["vinho"] = {\
    "data":  ('wine-recognition.tsv','\t'), \
    "types": ('wine-recognition_types.csv', ';')}

# ---------------------------------------------------------------------


def main():
    if len(sys.argv) > 1 and sys.argv[1] in dataset.keys():
        ds = dataset[sys.argv[1]]
    else: 
        ds = dataset["votos"] # default dataset 

    df_train = pd.read_csv(ds["data"][0], sep=ds["data"][1])
    df_train_attribute = pd.read_csv(ds["types"][0], sep=ds["types"][1])


    key_list=[]
    type_list=[]
    for i in range(df_train_attribute.shape[0]):
        #print(df_train_attribute.values[i])
        key_list.append(df_train_attribute.values[i][0])
        type_list.append(df_train_attribute.values[i][1])

#    print(key_list)
    
    target_attribute = key_list[-1]

#    print(target_attribute)

    attr_type_dict = dict(zip(key_list, type_list))
    df_train = df_train.astype(attr_type_dict)



    arvore = Arvore(df_train, target_attribute, ID3, len(key_list)-1)
    arvore.print()

#    print("teste de uma instancia:")
#    for j, test_row in df_train.iterrows():
#        print(test_row)
#        print("predicao: ", arvore.predict(test_row))
#        break
    


    #arvore.predict()

if __name__ == "__main__" :
    main()

