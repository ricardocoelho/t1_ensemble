import pandas as pd
import math

def get_info(df, target_attr):
    info = 0
    for c, df_split in df.groupby(target_attr):
        p= len(df_split.index)/len(df.index)
        info -= p*math.log2(p)
    return info

def ID3(df, target_attr):
    D= get_info(df, target_attr)
#    print("D :{}".format(D))
    info_attr={}
    df_len = len(df.index)

    for col in df.drop(columns=target_attr).columns:   #para todas as colunas exceto o atributo alvo:
        info_attr[col]=0
        for value, df_split in df.groupby(col):        #divide o df pelos valores do atributo
            val_occurences= len(df_split.index)
            pv = val_occurences/df_len                 #probabilidade do valor do atributo sobre todo o conjunto
#            print("{}: {}= {}/{}={}.".format(col, value, val_occurences, df_len, pv))
            info = get_info(df_split, target_attr)
#            print("info: {}".format(info))
            info_attr[col] += pv*info
        info_attr[col]  = D - info_attr[col]

    #escolhe o atributo com maior ganho de informacao
    max_val=-1;
    for k,v in info_attr.items():
#        print("ID3: {}: {} bits".format(k,v))
        if(v > max_val):
            max_val = v
            max_attr = k
    return max_attr, max_val


