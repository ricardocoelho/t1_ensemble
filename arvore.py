import math
import pandas as pd
from node import Node
from ID3 import ID3


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
    node= Node();
    most_freq_val = df[target_attr].value_counts().idxmax()

    node.set_category(most_freq_val)            #guarda valor mais frequente do atributo alvo
    

    for value, sub_df in df.groupby(target_attr):
        if len(sub_df.index) == len(df.index):   #tamanho do df dividido eh o mesmo que do original, homogeneo
            node.set_leaf()
            return node

#    if df[target_attr].value_counts().size == 1:  #atributo alvo homogeneo
#        node.set_leaf()
#        return node

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


def main():
    df_train = pd.read_csv('dadosBenchmark_validacaoAlgoritmoAD.csv', sep=';')
    print(df_train)
    print(df_train.dtypes)
    
# ****necessario somente se os atributos numericos nao forem identificados como 'int64'

#    #adiciona informações para o tipo de cada atributo(categorico/continuo
#    key_list = ['Tempo', 'Temperatura', 'Umidade', 'Ventoso', 'Joga']
#    type_list = ['category','category', 'category', 'category',  'category']
#    attr_type_dict = dict(zip(key_list, type_list))
#    df_train = df_train.astype(attr_type_dict)
#    print(df_train.dtypes)


    #gera a arvore
    arvore = create_tree(df_train, df_train.columns[-1], ID3)

    print_tree(arvore,0, "")

    #testa instancia
    uma_instancia= df_train[-1:]
    value = arvore.predict(uma_instancia)
    print("predicao teste:", value);


if __name__ == "__main__" :
    main()

