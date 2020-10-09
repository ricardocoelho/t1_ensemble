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

def bootstrap_table(df_train, key_list, attr_type_dict):
    #print(df_train.values[4])
    #seleciona conjunto de treino
    bs_list=[]
    n_dados_teste=df_train.shape[0]
    escolhidos=random.choices(range(0, n_dados_teste), k=n_dados_teste)
    for i in range(n_dados_teste):
        bs_list.append(df_train.values[escolhidos[i]])

    bs_table=pd.DataFrame.from_records(bs_list, columns=key_list)
    bs_table = bs_table.astype(attr_type_dict)
    #print("\nBOOTSTRAP\n",bs_table)   

    #para conferir que é amostragem com reposição
    #bs_table_unica=pd.DataFrame.drop_duplicates(bs_table)
    #print("\ntabela valores unicos\n",bs_table_unica)
    return bs_table, escolhidos

def out_of_bag_table(df_train, escolhidos, key_list, attr_type_dict):
    #seleciona conjunto de teste
    n_escolhidos=[]
    for i in range(df_train.shape[0]):
        if i not in escolhidos:
            n_escolhidos.append(i)

    out_list=[]
    for i in range(len(n_escolhidos)):
        out_list.append(df_train.values[n_escolhidos[i]])

    out_table=pd.DataFrame.from_records(out_list, columns=key_list)
    out_table = out_table.astype(attr_type_dict)
    #print("NAO ESCOLHIDOS\n",out_of_bag_table)

    return out_table, n_escolhidos

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
    #print(df_train_attribute)
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

    bootstrap, escolhidos = bootstrap_table(df_train, key_list, attr_type_dict)
    out_of_bag, n_escolhidos = out_of_bag_table(df_train, escolhidos, key_list, attr_type_dict)

    #gera a arvore
    arvore = create_tree(bootstrap, bootstrap.columns[-1], ID3)

    print_tree(arvore,0, "")

    #testa instancia
    
    #uma_instancia= df_train[-1:]
    for i in range(out_of_bag.shape[0]):
        uma_instancia=out_of_bag[i:i+1]
        print(uma_instancia)
        value = arvore.predict(uma_instancia)
        print("predicao teste:", value);


if __name__ == "__main__" :
    main()

