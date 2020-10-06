import math
import pandas as pd
from node import Node
from ID3 import ID3


def print_tree(node, i, edge):
    if node.is_leaf:
        print("[{}]{}{} {}".format(i, " "*i*4, edge, node.category))
    else:
        print("[{}]{}{} '{}'(gain: {:.3f} bits)".format(i, " "*i*4, edge, node.attribute, node.gain, ))
        for key, val in node.child.items():
            print_tree(val, i+1, key+" ->")


def create_tree(df, target_attr, selection_algorithm):
    node= Node();
    most_freq_val = df[target_attr].value_counts().idxmax()

    node.set_category(most_freq_val)            #guarda valor mais frequente do atributo alvo
    
    if df[target_attr].value_counts().size == 1:  #atributo alvo homogeneo
        node.set_leaf()
        return node

    if df.shape[1] == 1: #nao há novos atributos
        node.set_leaf()
        return node

    attr, gain = selection_algorithm(df, target_attr)
#    print('choosen attribute: {}'.format(attr))

    node.set_attribute(attr)
    node.set_gain(gain)

    for attr_val, df_splited in df.groupby(attr): #divide as instancias pelos valores do atributo selecionado
        node.child[attr_val] = create_tree(df_splited.drop(columns=attr), target_attr, selection_algorithm)

    return node



def main():
    df_train = pd.read_csv('dadosBenchmark_validacaoAlgoritmoAD.csv', sep=';')

    print(df_train)

    #gera a arvore
    arvore = create_tree(df_train, df_train.columns[-1], ID3)

    print_tree(arvore,0, "")

    #testa instancia
    uma_instancia= df_train[-1:]
    value = arvore.predict(uma_instancia)
    print("predicao teste:", value);


if __name__ == "__main__" :
    main()

