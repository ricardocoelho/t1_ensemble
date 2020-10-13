import sys
import math
import pandas as pd
import random
import numpy as np
from arvore import Arvore, FlorestaAleatoria
# ---------------------------------------------------------------------
dataset = {}
dataset["votos"] = {\
    "data": ('house-votes-84.tsv', '\t'), \
    "types": ('house-votes-84_types.csv', ';')}
dataset["jogo"] =  { \
    "data": ('dadosBenchmark_validacaoAlgoritmoADv2.csv', ';'), \
    "types": ('AttributeType.csv', ';')}
dataset["vinho"] = {\
    "data":  ('wine-recognition.tsv','\t'), \
    "types": ('wine-recognition_types.csv', ';')}

# ---------------------------------------------------------------------

def cross_validation(df, target, K):
    target_classes = df[target].value_counts() # TODO: unused var
    #dividir o dataset nas classes do atributo alvo
    df_list= []

    for name, df in df.groupby(target):
        df_list.append(df)

    print([len(x) for x in df_list])
    #tamanho de cada fold para negativos e positivos
    fold_size_per_target = list(map(lambda x: round(len(x.index)/K), df_list))

    print(fold_size_per_target)

    #divide em K folds
    fold_list_per_target=[]

    for df, fold_size in zip(df_list, fold_size_per_target):
        #print("ashdasu")
        fold_class_list =[]
        for i in range(0,K):
            if(i==K-1):
                fold_class_list.append(df[fold_size*i :])
            else:
                fold_class_list.append(df[fold_size*i : fold_size*(i+1)])

        fold_list_per_target.append(fold_class_list)


    #junta os K folds em uma lista unica, contendo folds estratificados
    fold_list= []
    for fold in zip(*fold_list_per_target):
        fold_list.append(pd.concat(list(fold), axis=0))

    print(fold_list)

    table_of_confusion_list= [None for _ in range(K)]

    target_coluna = list(df.columns).index(target)

    for i in range(0,K):
        train = pd.concat(fold_list[:i] + fold_list[i+1:], axis=0)
        test = fold_list[i]
        n_arvores=3
        modelo = FlorestaAleatoria(train.copy(), target_coluna, n_arvores); #parametro n_arvores


        #inicializa a matriz de confusao
        table_of_confusion_list[i]= {'CERTO':0,'ERRADO':0}

        print("TESTE")
        #roda o KNN para cada instancia do fold de teste atual
        #test.reset_index(drop=True)
        ### (the following two lines work to reset j but i disapprove)
        ### test = test.copy() 
        ### test.index = range(len(test.index))
        nrow = 0
        for j, test_row in test.iterrows():
            
            resp= modelo.predict(test_row)
            print("[{}/{}]{:0.2f}% complete...".format(i+1, K,100*(nrow+1)/len(test.index) ), end='\r')
            nrow += 1
            #atualiza a matriz de confusao
            if resp == test_row[target]:
                table_of_confusion_list[i]['CERTO'] += 1
            else:
                table_of_confusion_list[i]['ERRADO'] += 1

    print()
    print(table_of_confusion_list)

    acuracia_total=0
    total_testes=0
    for item in table_of_confusion_list:
        acuracia_total=acuracia_total+item['CERTO']
        total_testes=total_testes+item['CERTO']+item['ERRADO']
    acuracia_total=acuracia_total/total_testes
    print("n: ",n_arvores," Acuracia: ",acuracia_total)

# ---------------------------------------------------------------------
def main():
    random.seed(10)
    np.random.seed(10)

    if len(sys.argv) > 1 and sys.argv[1] in dataset.keys():
        ds = dataset[sys.argv[1]]
    else: 
        ds = dataset["votos"] # default dataset 

    df_train = pd.read_csv(ds["data"][0], sep=ds["data"][1])
    df_train_attribute = pd.read_csv(ds["types"][0], sep=ds["types"][1])

    print(df_train.head(5))
    #target_coluna=0     #<<<<<<<<<<<-----------------coluna onde está o target
    
    key_list=[]
    type_list=[]
    for i in range(df_train_attribute.shape[0]):
        #print(df_train_attribute.values[i])
        key_list.append(df_train_attribute.values[i][0])
        type_list.append(df_train_attribute.values[i][1])

    print(key_list)
    
    target_attribute = key_list[-1]

    print(target_attribute)

    attr_type_dict = dict(zip(key_list, type_list))
    df_train = df_train.astype(attr_type_dict)
    print(df_train.dtypes)

    cross_validation(df_train, target_attribute, 10)

    return
# ---------------------------------------------------------------------

    # modelo = FlorestaAleatoria(df_train.copy(), target_coluna, 11); #parametro n_arvores


# #################################teste
#     #testa instancia
#     #por enquanto, vai pegar as 10 primeiras linhas do conjunto de treino
#     #uma_instancia= df_train[-1:]
#     print("\n\nTESTES:")

#     for i in range(0,10):
#         uma_instancia=df_train[i:i+1]
#         real_value = uma_instancia.iloc[0][uma_instancia.columns[target_coluna]]

#         predicted_value = modelo.predict(uma_instancia)
#         print("real, predito: ({}, {})".format(real_value , predicted_value));


if __name__ == "__main__" :
    main()

