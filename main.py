import math
import pandas as pd
import random
import numpy as np
from arvore import Arvore, FlorestaAleatoria

def main():
    random.seed(10)
    np.random.seed(10)

#    df_train = pd.read_csv('dadosBenchmark_validacaoAlgoritmoADv2.csv', sep=';')
#    df_train = pd.read_csv('house-votes-84.tsv', sep='\t')
    df_train = pd.read_csv('wine-recognition.tsv', sep='\t')
    print(df_train.head(5))
    target_coluna=0#<<<<<<<<<<<-----------------coluna onde está o target
    
#    #adiciona informações para o tipo de cada atributo(categorico/continuo
#    df_train_attribute = pd.read_csv('AttributeType.csv', sep=';')
    df_train_attribute = pd.read_csv('wine-recognition_types.csv', sep=';')

    key_list=[]
    type_list=[]
    for i in range(df_train_attribute.shape[0]):
        #print(df_train_attribute.values[i])
        key_list.append(df_train_attribute.values[i][0])
        type_list.append(df_train_attribute.values[i][1])

    print(key_list)
    

    attr_type_dict = dict(zip(key_list, type_list))
    df_train = df_train.astype(attr_type_dict)
    print(df_train.dtypes)


    modelo = FlorestaAleatoria(df_train.copy(), target_coluna, 11); #parametro n_arvores


#################################teste
    #testa instancia
    #por enquanto, vai pegar as 10 primeiras linhas do conjunto de treino
    #uma_instancia= df_train[-1:]
    print("\n\nTESTES:")

    for i in range(0,10):
        uma_instancia=df_train[i:i+1]
        real_value = uma_instancia.iloc[0][uma_instancia.columns[target_coluna]]

        predicted_value = modelo.predict(uma_instancia)
        print("real, predito: ({}, {})".format(real_value , predicted_value));




if __name__ == "__main__" :
    main()

