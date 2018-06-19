from collections import OrderedDict
from pandas import Series, DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
top_labels = []


def combine_files(files):
    frames = files
    result = pd.concat(frames, ignore_index = True)
    print(result)
    result.to_csv('/Users/Amelia/Desktop/combined_files.csv', index_label='Index')


def cate_to_dummy(input_data,full_set=False):
    """
    usage : Given a full_set, this function converts a DataFrame of several columns into a
            DataFrame with each column being a dummy variable of an element in full_set.
            If the full_set argument is not given, the function will produce a full_set by
            finding the union of the columns in input_data.
    """
    ncol=input_data.columns.size
    nrow=input_data.index.size
    if full_set is False:
        print("in if loop")
        full_set=[]
        for i in range(ncol):
            print("in for loop")
            full_set=list(set(full_set).union(set(input_data[input_data.columns[i]])))
        full_set.sort()
    df_return=DataFrame(np.zeros([nrow,len(full_set)]).astype(int),
                        columns=full_set,index=input_data.index)
    for i in range(nrow):
        print(i)
        for j in range(ncol):
            print(j)
            df_return.iloc[i][input_data.iloc[i][input_data.columns[j]]]=1
    return(df_return)


def read_file(filename):
    # 25622 rows  '
    df = pd.read_csv(filename)
    # 25344 rows
    # get diagnosis '465'
    df1 = df.loc[df['DIAG13D'] != '000']
    df1 = df.loc[df['DIAG13D'] == '465']
    dias = []
    all_rfv = df[['RFV1','RFV2','RFV3']]
    df2 = cate_to_dummy(all_rfv, False)

    df3 = pd.concat([df1, df2], axis=1)

    all_drugs = pd.concat([df1['DRUGID1'], df1['DRUGID2'], df1['DRUGID3'], df1['DRUGID4'],
                           df1['DRUGID5'], df1['DRUGID6'], df1['DRUGID7'], df1['DRUGID8']])
    # select top  drugs
    dicts_drugs = all_drugs.value_counts()
    dict(dicts_drugs)
    sorted_dicts_drugs = sorted(dicts_drugs.iteritems(), key=lambda x: x[1], reverse=True)
    sorted_dicts_drugs.pop(0)
    target = []

    for each in range(80):
            target.append(sorted_dicts_drugs[each][0])

    global top_labels
    top_labels = target
    df4 = pd.DataFrame((np.zeros(len(df3) * len(target), dtype=np.int).reshape(len(df3), len(target))), index=df3.index,
                       columns=target)

    num = list(df3.index)
    for each in num:
        a = df3.loc[each, 'DRUGID1']
        if a in target:
            df4.loc[each, a] = 1
        b=df3.loc[each, 'DRUGID2']
        if b in target:
            df4.loc[each, b] = 1
        c=df3.loc[each, 'DRUGID3']
        if c in target:
            df4.loc[each, c] = 1
        d=df3.loc[each, 'DRUGID4']
        if d in target:
            df4.loc[each, d] = 1
        e=df3.loc[each, 'DRUGID5']
        if e in target:
            df4.loc[each,e] = 1
        f=df3.loc[each, 'DRUGID6']
        if f in target:
            df4.loc[each,f] = 1
        g=df3.loc[each, 'DRUGID7']
        if f in target:
            df4.loc[each,f] = 1
        h=df3.loc[each, 'DRUGID8']
        if f in target:
            df4.loc[each,f] = 1

    df5 = pd.concat([df4, df3], axis=1)
    df5 = df5.reset_index(drop=True)
    df5['normalAge'] = StandardScaler().fit_transform(df5['AGE'].reshape(-1, 1))
    df5 = df5.drop(['AGE','DIAG13D','DRUGID8','DRUGID7', 'DRUGID6', 'DRUGID5', 'DRUGID4', 'DRUGID3',
                  'DRUGID2','DRUGID1','RFV1','RFV2','RFV3'], axis=1)
    df5.to_csv(filename.split('.')[0] + '_multidrug_AH.csv')


def add_feature(filename):
    df = pd.read_csv(filename)
    df1 = df
    col = df.columns
    list1 = ['etoh','breast','depress','dvs','foot','neuro','pelvic',\
             'rectal','retinal','skin','subst','bmp','cbc','chlamyd','cmp','creat','bldcx','trtcx','urncx',\
             'othcx','glucose','gct', 'hgba','heptest','hivtest','hpvdna','cholest','hepatic','pap',\
             'pregtest','psa','strep','thyroid','urine','vitd','bonedens','catscan','echocard','othultra','mammo','mri','xray','audio',\
             'biopsy','cardiac','colon','cryo','ekg','eeg','emg','excision','fetal','peak','sigmoid','spiro','tono','tbtest','egd','sigcolon']
    all_old_disa = [x.upper() for x in list1]
    for each in all_old_disa:
        if each not in col:
            df1[each] = 2


    df2 = df1[all_old_disa]
    print(df2)
    print(df2.shape)
    df3 = df[['AGE','SEX','RFV1','RFV2','RFV3',
            'DIAG13D','DRUGID8','DRUGID7', 'DRUGID6', 'DRUGID5',
            'DRUGID4', 'DRUGID3','DRUGID2','DRUGID1']]
    df4 = pd.concat([df3, df2], axis=1)
    print(df4)
    print(df4.shape)
    df4.to_csv(filename.split('.')[0] + '_muti.csv')


# def find_freq(filename):
#     df = pd.read_csv(filename)
#     # scaling to unit variance
#     df = df.drop(['Unnamed: 0', 'AGE','SEX','MENTSTAT', 'BLODPRES', 'EKG', 'CARDMON', 'PULSOXIM', 'URINE',
#                   'PREGTEST', 'HIVSER', 'BLOODALC', 'CBC','CHESTXRY', 'PROC', 'MRI', 'ULTRASND',
#                   'CATSCAN', 'PROC', 'CPR', 'ENDOINT', 'CPR', 'IVFLUIDS', 'BLADCATH', 'WOUND', 'EYEENT',
#                   'ORTHOPED', 'OBGYN','DIAG13D','DRUGID8','DRUGID7', 'DRUGID6', 'DRUGID5', 'DRUGID4', 'DRUGID3',
#                   'DRUGID2','DUGID1'], axis=1)
#     freq = {}
#     sum_elements = len(df)
#     for column in df:
#         freq[column] = sum(df[column])/sum_elements
#     df2 = pd.DataFrame([freq], columns=freq.keys())
#     print(df2)


def main():
    # df1 = pd.read_csv('/Users/Amelia/Desktop/newfiles/OPD2010_muti.csv')
    # df2 = pd.read_csv('/Users/Amelia/Desktop/newfiles/opd2011_muti.csv')
    # df3 = pd.read_csv('/Users/Amelia/Desktop/newfiles/ed2014_muti.csv')
    # df4 = pd.read_csv('/Users/Amelia/Desktop/newfiles/ed2015-spss_muti.csv')
    # df5 = pd.read_csv('/Users/Amelia/Desktop/newfiles/NAMCS2014_muti.csv')
    # df6 = pd.read_csv('/Users/Amelia/Desktop/newfiles/NAMCS2015_muti.csv')
    # files = [df1,df2,df3,df4,df4,df6]
    # combine_files(files)
    read_file('/Users/Amelia/Desktop/combined_files.csv')
    # add_feature('/Users/Amelia/Desktop/newfiles/OPD2010.csv')
    #add_feature('/Users/Amelia/Desktop/newfiles/opd2011.csv')
    #add_feature('/Users/Amelia/Desktop/newfiles/ed2014.csv')
    #add_feature('/Users/Amelia/Desktop/newfiles/ed2015-spss.csv')
    #add_feature('/Users/Amelia/Desktop/newfiles/NAMCS2014.csv')
    #add_feature('/Users/Amelia/Desktop/newfiles/NAMCS2015.csv')
    # classify('/Users/Amelia/Desktop/newfiles/ED00_multidrug_AH.csv')

if __name__ == '__main__':
        main()