import math
import pandas as pd
from node import Node
from ID3 import ID3
import random

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


def create_tree(df, target_attr, selection_algorithm):
    node= Node()
    most_freq_val = df[target_attr].value_counts().idxmax()

    node.set_category(most_freq_val)            #guarda valor mais frequente do atributo alvo
    

    for value, sub_df in df.groupby(target_attr):
        if len(sub_df.index) == len(df.index):   #tamanho do df dividido eh o mesmo que do original, homogeneo
            node.set_leaf()
            return node

    if df.shape[1] == 1: #nao há novos atributos
        node.set_leaf()
        return node

    attr, gain = selection_algorithm(df, target_attr)
#    print('choosen attribute: {}'.format(attr))

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
            node.child[attr_val] = create_tree(df_splited.drop(columns=attr), target_attr, selection_algorithm)
        
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

    new_key_list=[]
    #print("São ",n_atri," atributos")
    escolhidos=random.sample(range(0, n_atri-1), k=int(n_atri ** (1/2)))#amostragem sem reposição
    for i in range(int(n_atri ** (1/2))):
        #print(i," escolhido ",escolhidos[i],"esse",key_list[escolhidos[i]],"e esse",type_list[escolhidos[i]])
        new_key_list.append(key_list[escolhidos[i]])
    new_key_list.append(key_list[-1])


    print("atributos sorteados", new_key_list)
    return new_key_list


def main():
    df_train = pd.read_csv('dadosBenchmark_validacaoAlgoritmoADv2.csv', sep=';')
    print(df_train)
    print(df_train.dtypes)
    
# ****necessario somente se os atributos numericos nao forem identificados como 'int64'

#    #adiciona informações para o tipo de cada atributo(categorico/continuo
#    key_list = ['Tempo', 'Temperatura', 'Umidade', 'Ventoso', 'Joga']
#    type_list = ['category','category', 'category', 'category',  'category']
#    attr_type_dict = dict(zip(key_list, type_list))
#    df_train = df_train.astype(attr_type_dict)
#    print(df_train.dtypes)
    df_train_attribute = pd.read_csv('AttributeType.csv', sep=';')

    print(df_train_attribute)
    #print(df_train_attribute.dtypes)

    key_list=[]
    type_list=[]
    for i in range(df_train_attribute.shape[0]):
        #print(df_train_attribute.values[i])
        key_list.append(df_train_attribute.values[i][0])
        type_list.append(df_train_attribute.values[i][1])

    print("Os atributos são ",key_list," com tipos ",type_list)
    attr_type_dict = dict(zip(key_list, type_list))
    df_train = df_train.astype(attr_type_dict)
    print(df_train.dtypes)
#################################começa a geração das árvores

    new_key_list = amostragem_atributos(key_list)
    #abre o arquivo somente com as colunas selecionadas

    new_dt_train = df_train[new_key_list]
    print(new_dt_train)

    #seleciona conjuntos de treinamento e teste
    bootstrap = bootstrap_table(new_dt_train)
    print("bootstap")
    print(bootstrap)

    out_of_bag = out_of_bag_table(new_dt_train, bootstrap)
    print("out of bag")
    print(out_of_bag)

    #gera a arvore
    arvore = create_tree(bootstrap, bootstrap.columns[-1], ID3)

    print_tree(arvore,0, "")

#################################teste
    #testa instancia
    
    #uma_instancia= df_train[-1:]
    print("TESTES:")
    for i in range(out_of_bag.shape[0]):
        uma_instancia=out_of_bag[i:i+1]
        print(uma_instancia)
        value = arvore.predict(uma_instancia)
        print("predicao teste:", value);

    return


if __name__ == "__main__" :
    main()

