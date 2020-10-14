import sys
import math
import pandas as pd
import random
import numpy as np
from arvore import Arvore, FlorestaAleatoria
from multiprocessing import Pool

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
n_arvores=5


#def treina_e_testa(treino, teste, target_coluna, n_arvores):
def treina_e_testa(args):
    treino , teste, target_coluna, n_arvores = args
    modelo = FlorestaAleatoria(treino, target_coluna, n_arvores); #parametro n_arvores
    key_list = list(treino.columns)
    target_name= key_list[target_coluna]

    #inicializa a matriz de confusao
    table_of_confusion = {'CERTO':0,'ERRADO':0}

    nrow = 0
    for j, test_row in teste.iterrows():
        
        resp= modelo.predict(test_row)
        #print("[{}/{}]{:0.2f}% complete...".format(i+1, K,100*(nrow+1)/len(test.index) ), end='\r')
        nrow += 1
        #atualiza a matriz de confusao
        if resp == test_row[target_name]:
            table_of_confusion['CERTO'] += 1
        else:
            table_of_confusion['ERRADO'] += 1

    return table_of_confusion



def cross_validation(df, target, K):
    target_classes = df[target].value_counts() # TODO: unused var
    #dividir o dataset nas classes do atributo alvo
    df_list= []

    for name, df in df.groupby(target):
        df_list.append(df)

#    print("numero de instancias por classe: ", [len(x) for x in df_list])
    #tamanho de cada fold para negativos e positivos
    fold_size_per_target = list(map(lambda x: round(len(x.index)/K), df_list))

#    print(fold_size_per_target)

    #divide em K folds
    fold_list_per_target=[]

    for df, fold_size in zip(df_list, fold_size_per_target):
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

#    print(fold_list)

    target_coluna = list(df.columns).index(target)

    
    train = [None]*K
    test = [None]*K
    #monta os folds de treino e teste
    for i in range(K):
        train[i] = pd.concat(fold_list[:i] + fold_list[i+1:], axis=0).copy()
        test[i] = fold_list[i].copy()

    #roda os treinos em paralelo
    with Pool(processes=4) as pool:
        #encapsula os argumentos
        arg_list = (zip(train, test, [target_coluna]*K, [n_arvores]*K))
        result_list = pool.map(treina_e_testa, arg_list)

#    print(result_list)

    #calculo das metricas
    acc = [None]*len(result_list)
    for i in range (len(result_list)):
        total = result_list[i]['CERTO'] + result_list[i]['ERRADO']
        acc[i] = result_list[i]['CERTO']/total

    acuracia_media = np.average(acc)
    desvio_padrao = np.std(acc)

    return n_arvores, acuracia_media, desvio_padrao

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

#    print(df_train.head(5))
    #target_coluna=0     #<<<<<<<<<<<-----------------coluna onde está o target
    
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
#    print(df_train.dtypes)

    global n_arvores

    valores_de_teste = [1,5,11,25]  #valores para n_arvores
    
    print("n; Acuracia; desvio_padrao")

    for n in valores_de_teste:
        n_arvores = n
        n, acc, stdev = cross_validation(df_train, target_attribute, 10)
        print("{}; {}; {}".format(n, acc, stdev))
    return


if __name__ == "__main__" :
    main()

