from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
top_labels = []

def read_file(filename):
    # 25622 rows  '
    df = pd.read_csv(filename, usecols=['AGE','SEX','RFV1','RFV2','RFV3','MENTSTAT', 'BLODPRES', 'EKG', 'CARDMON', 'PULSOXIM', 'URINE',
                                        'PREGTEST', 'HIVSER', 'BLOODALC', 'CBC','CHESTXRY', 'PROC', 'MRI', 'ULTRASND',
                                        'CATSCAN', 'PROC', 'CPR', 'ENDOINT', 'CPR', 'IVFLUIDS', 'BLADCATH', 'WOUND', 'EYEENT',
                                        'ORTHOPED', 'OBGYN','DIAG13D','GEN1','GEN2','GEN3','GEN4','GEN5','GEN6'])
    # 25344 rows
    # get diagnosis '465'
    df1 = df.loc[df['DIAG13D'] != '000']
    df1 = df.loc[df['DIAG13D'] == '465']
    all_drugs = pd.concat([df1['GEN1'], df1['GEN2'], df1['GEN3'], df1['GEN4'],
                           df1['GEN5'], df1['GEN6']])
    df3=pd.get_dummies(df1,columns=['RFV1','RFV2','RFV3'])
    # select top 15 drugs
    dicts_drugs = all_drugs.value_counts()
    dict(dicts_drugs)
    sorted_dicts_drugs = sorted(dicts_drugs.iteritems(), key=lambda x: x[1], reverse=True)

    target = []

    for each in range(15):
        target.append(sorted_dicts_drugs[each][0])

    global top_labels
    top_labels = target
    df2 = pd.DataFrame((np.zeros(len(df1) * len(target), dtype=np.int).reshape(len(df1), len(target))), index=df1.index,
                       columns=target)

    num = list(df1.index)
    for each in num:
        a = df1.loc[each, 'GEN1']
        if a in target:
            df2.loc[each, a] = 1
        b=df1.loc[each, 'GEN2']
        if b in target:
            df2.loc[each, b] = 1
        c=df1.loc[each, 'GEN3']
        if c in target:
            df2.loc[each, c] = 1
        d=df1.loc[each, 'GEN4']
        if d in target:
            df2.loc[each, d] = 1
        e=df1.loc[each, 'GEN5']
        if e in target:
            df2.loc[each,e] = 1
        f=df1.loc[each, 'GEN6']
        if f in target:
            df2.loc[each,f] = 1

    df4 = pd.concat([df3, df2], axis=1)
    df4.head()
    df4 = df4.reset_index(drop=True)
    df4.to_csv(filename.split('.')[0] + '_multidrug_AH.csv')


def add_feature(filename):
    df = pd.read_csv(filename)
    col=df.columns
    list1 = ['etoh','breast','depress','dvs','foot','neuro','pelvic',\
             'rectal','retinal','skin','subst','bmp','cbc','chlamyd','cmp','creat','bldcx','trtcx','urncx',\
             'othcx','glucose','gct', 'hgba','heptest','hivtest','hpvdna','cholest','hepatic','pap',\
             'pregtest','psa','strep','thyroid','urine','vitd','bonedens','catscan','echocard','othultra','mammo','mri','xray','audio',\
             'biopsy','cardiac','colon','cryo','ekg','eeg','emg','excision','fetal','peak','sigmoid','spiro','tono','tbtest','egd','sigcolon']
    all_old_disa = [x.upper() for x in list1]
    for each in all_old_disa:
        if each not in col:
            df[each]=2

    df['normalAge']=StandardScaler().fit_transform(df['AGE'].reshape(-1,1))
    df = df.drop(['Unnamed: 0','GEN2','GEN3','GEN4','GEN5','GEN6','AGE'], axis=1)
    df.to_csv(filename.split('.')[0] + '_final.csv')


def find_freq(filename):
    df = pd.read_csv(filename)
    # scaling to unit variance
    df = df.drop(['Unnamed: 0', 'AGE','SEX','MENTSTAT', 'BLODPRES', 'EKG', 'CARDMON', 'PULSOXIM', 'URINE',
                  'PREGTEST', 'HIVSER', 'BLOODALC', 'CBC','CHESTXRY', 'PROC', 'MRI', 'ULTRASND',
                  'CATSCAN', 'PROC', 'CPR', 'ENDOINT', 'CPR', 'IVFLUIDS', 'BLADCATH', 'WOUND', 'EYEENT',
                  'ORTHOPED', 'OBGYN','DIAG13D','GEN1','GEN2','GEN3','GEN4','GEN5','GEN6'], axis=1)
    freq = {}
    sum_elements = len(df)
    for column in df:
        freq[column] = sum(df[column])/sum_elements
    df2 = pd.DataFrame([freq], columns=freq.keys())
    print(df2)





def main():
    read_file('/Users/Amelia/Desktop/newfiles/ED00.csv')
    add_feature('/Users/Amelia/Desktop/newfiles/ED00_multidrug_AH.csv')
    # classify('/Users/Amelia/Desktop/newfiles/ED00_multidrug_AH.csv')

if __name__ == '__main__':
        main()