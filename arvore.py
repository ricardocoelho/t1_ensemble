import math
import pandas as pd
from node import Node
from ID3 import ID3
import random
import numpy as np

def print_tree(node, i, edge):
    if node.is_leaf:
        print("[{}]{}{} {}".format(i, " "*i*4, edge, node.category))
    else:
        node_name = node.attribute
        if node.numeric_condition:
            node_name += " <= " + "{:.2f}".format(node.numeric_condition)
        print("[{}]{}{} '{}'(gain: {:.3f} bits)".format(i, " "*i*4, edge, node_name , node.gain, ))
        for key, val in node.child.items():
            print_tree(val, i+1, str(key)+" ->")


def create_tree(df, target_attr, selection_algorithm, m=None):
    if not m:
        m = len(df.columns) - 1

    node= Node()
    most_freq_val = df[target_attr].value_counts().idxmax()

    node.set_category(most_freq_val)            #guarda valor mais frequente do atributo alvo

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

    sampled_attr_list = amostragem_atributos(attr_list)


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
    #print(treino.values[4])
    #seleciona conjunto de treino
#    bs_list=[]

    n_dados_teste = treino.shape[0]
#
#    escolhidos = random.choices(range(0, n_dados_teste), k=n_dados_teste)
#    for i in range(n_dados_teste):
#        bs_list.append(treino.values[escolhidos[i]])
#
#    bs_table=pd.DataFrame.from_records(bs_list, columns=chaves_list)
#    bs_table = bs_table.astype(atri_dict)


    return treino.sample(n=n_dados_teste, replace = True)

    #print("\nBOOTSTRAP\n",bs_table)   

    #para conferir que é amostragem com reposição
    #bs_table_unica=pd.DataFrame.drop_duplicates(bs_table)
    #print("\ntabela valores unicos\n",bs_table_unica)
    return bs_table

def out_of_bag_table(df_train, escolhidos):
    #seleciona conjunto de teste
#    n_escolhidos=[]
#    for i in range(df_train.shape[0]):
#        if i not in escolhidos:
#            n_escolhidos.append(i)
#
#    out_list=[]
#    for i in range(len(n_escolhidos)):
#        out_list.append(df_train.values[n_escolhidos[i]])
#
#    out_table=pd.DataFrame.from_records(out_list, columns=df.train.columns)
#    out_table = out_table.astype(attr_type_dict)
#    #print("NAO ESCOLHIDOS\n",out_of_bag_table)

    return df_train[~df_train.index.isin(escolhidos.index)]

def amostragem_atributos(key_list):
    n_atri=len(key_list)
    num_samples = int(n_atri ** (1/2))
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


def main():
    random.seed(10)
    np.random.seed(10)

#    df_train = pd.read_csv('dadosBenchmark_validacaoAlgoritmoADv2.csv', sep=';')
    df_train = pd.read_csv('house-votes-84.tsv', sep='\t')
    print(df_train.head(5))
    
#    #adiciona informações para o tipo de cada atributo(categorico/continuo
#    df_train_attribute = pd.read_csv('AttributeType.csv', sep=';')
    df_train_attribute = pd.read_csv('house-votes-84_types.csv', sep=';')

    key_list=[]
    type_list=[]
    for i in range(df_train_attribute.shape[0]):
        #print(df_train_attribute.values[i])
        key_list.append(df_train_attribute.values[i][0])
        type_list.append(df_train_attribute.values[i][1])

    attr_type_dict = dict(zip(key_list, type_list))
    df_train = df_train.astype(attr_type_dict)
    print(df_train.dtypes)

#################################começa a geração das árvores
    floresta=[]
    n_arvores=3
    for i in range(n_arvores):
        #seleciona conjuntos de treinamento e teste
        bootstrap = bootstrap_table(df_train.copy())
        print("BOOTSTRAP:")
        print(bootstrap)

        out_of_bag = out_of_bag_table(df_train.copy(), bootstrap)
        print("OUT OF BAG:")
        print(out_of_bag)

        #gera a arvore
        arvore = create_tree(bootstrap, bootstrap.columns[-1], ID3)
        floresta.append(arvore)
        print_tree(arvore,0, "")

#################################teste
    #testa instancia
    #por enquanto, vai pegar o out of bag do ultimo
    #uma_instancia= df_train[-1:]
    print("\n\nTESTES:")
    for i in range(out_of_bag.shape[0]):
        uma_instancia=out_of_bag[i:i+1]
        real_value = uma_instancia.iloc[0][uma_instancia.columns[-1]]
        votacao=0
        for arvore in floresta:
            predicted_value = arvore.predict(uma_instancia)
            votacao=votacao+predicted_value
        print(votacao,"votos")#, divide em ",int(n_arvores/2))
        if votacao <= int(n_arvores/2):
            predicted_value=0
        else:
            predicted_value=1
        print("real, predito: ({}, {})".format(real_value , predicted_value));

    return


if __name__ == "__main__" :
    main()

