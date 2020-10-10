import pandas as pd
import math
import numpy as np

#funcoes para calcular as metricas
def get_acc(table_of_confusion):
    VP, VN, FP, FN = table_of_confusion.values()
    return (VP+VN)/(VP+VN+FP+FN)

def get_rev(table_of_confusion):
    VP, VN, FP, FN = table_of_confusion.values()
    return VP/(VP+FN)

def get_prec(table_of_confusion):
    VP, VN, FP, FN = table_of_confusion.values()
    return VP/(VP+FP)

def get_f_measure(t_conf, b=1):
    VP, VN, FP, FN = t_conf.values()
    b_2 = math.pow(b,2)
    prec = get_prec(t_conf)
    rev = get_rev(t_conf)
    return (1+b_2)*(prec*rev)/(b_2*prec + rev)

def euclidean_dist(row, t_row):
    return math.sqrt(sum(map(lambda x,y: math.pow(x-y,2), row[:-1], t_row[:-1])))

def normalize_data(train_df, test_df):
    #normalizar todos os atributos exceto o ultimo:
    for col in train_df.columns[:-1]:
        max_val= max(train_df[col])
        min_val= min(train_df[col])
        train_df[col]= train_df[col].apply(lambda x: (x-min_val)/(max_val-min_val))
        test_df[col]= test_df[col].apply(lambda x: (x-min_val)/(max_val-min_val))
    return train_df, test_df


def KNN(train_df, test_inst, K=5):

    #aplica a distancia euclidiana em todas as linhas e salva na nova coluna 'Diff'
    train_df['Diff']= train_df.apply(euclidean_dist, t_row=test_inst, axis=1)

    #ordena e retorna novo dataframe com os K primeiros valores
    df_1 = train_df.sort_values(by='Diff').head(K)

    #retorna a classe da maioria
    return  int(sum(df_1['Outcome']==1) > sum(df_1['Outcome']==0))

def main():

    K=10        #numero de folds

    df = pd.read_csv('diabetes.csv')
    n_negative, n_positive = df['Outcome'].value_counts()

    #dividir o dataset em casos positivos e negativos
    neg = df['Outcome']==0
    pos = df['Outcome']==1
    df_n = df[neg].reset_index(drop=True)
    df_p = df[pos].reset_index(drop=True)

    #tamanho de cada fold para negativos e positivos
    fold_n_size = math.ceil(len(df_n.index)/K)
    fold_p_size = math.ceil(len(df_p.index)/K)

    #divide em K folds
    fold_n_list=[]
    for i in range(0,K):
        fold_n_list.append(df_n[fold_n_size*i : fold_n_size*(i+1)])

    fold_p_list=[]
    for i in range(0,K):
        fold_p_list.append(df_p[fold_p_size*i:fold_p_size*(i+1)])

    #junta os K folds negativo/positivo em uma lista unica, contendo folds estratificados
    fold_list=[]
    for fold_n, fold_p in zip(fold_n_list, fold_p_list):
        fold_list.append(pd.concat([fold_p, fold_n], axis=0).reset_index(drop=True))


    table_of_confusion_list= [None for _ in range(K)]

    for i in range(0,K):
        train = pd.concat(fold_list[:i] + fold_list[i+1:], axis=0).reset_index(drop=True)
        test = fold_list[i]

        #normaliza os dados
        train, test = normalize_data(train.copy(), test.copy())

        #inicializa a matriz de confusao
        table_of_confusion_list[i]= {'VP':0,'VN':0,'FP':0,'FN':0}

        #roda o KNN para cada instancia do fold de teste atual
        for j, test_row in test.iterrows():
            resp= KNN(train.copy(), test_row)
            print("[{}/{}]{:02.2f}% to complete...".format(i+1, K,100*j/len(test.index) ), end='\r')

            #atualiza a matriz de confusao
            if resp == test_row['Outcome']:
                if resp == 1:
                    table_of_confusion_list[i]['VP'] += 1
                else:
                    table_of_confusion_list[i]['VN'] += 1
            else:
                if resp == 1:
                    table_of_confusion_list[i]['FP'] += 1
                else:
                    table_of_confusion_list[i]['FN'] += 1

    print()

    acc = [None for _ in range(K)]
    rev = [None for _ in range(K)]
    prec = [None for _ in range(K)]
    f1 = [None for _ in range(K)]
    with open('out.csv','w') as f:
        f.write("{},{},{},{},{},{},{}\n".format('Iteration','Accuracy','F1_Measure','VP','VN','FP','FN'))

        #calcula as metricas para cada fold
        for i in range(len(table_of_confusion_list)):
            acc[i]= get_acc(table_of_confusion_list[i])
            rev[i]= get_rev(table_of_confusion_list[i])
            prec[i]= get_prec(table_of_confusion_list[i])
            f1[i]= get_f_measure(table_of_confusion_list[i], b=1)

            #print("{},{:.2f},{:.2f},{},{},{},{}".format(i, acc[i], f1[i], table_of_confusion_list[i]['VP'],
            #                                                                    table_of_confusion_list[i]['VN'],
            #                                                                table_of_confusion_list[i]['FP'],
            #                                                                table_of_confusion_list[i]['FN']))

            f.write("{},{:.2f},{:.2f},{},{},{},{}\n".format(i+1, acc[i], f1[i], table_of_confusion_list[i]['VP'],
                                                                                table_of_confusion_list[i]['VN'],
                                                                                table_of_confusion_list[i]['FP'],
                                                                                table_of_confusion_list[i]['FN']))
        f.write('Mean,{:.2f},{:.2f}\n'.format(sum(acc)/K, sum(f1)/K))
        f.write('Std_dev,{:.2f},{:.2f}\n'.format(np.std(acc), np.std(f1)))
            

if __name__ == "__main__" :
    main()

