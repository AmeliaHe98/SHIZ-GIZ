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
    df = pd.read_csv(filename, usecols=['VMONTH', 'AGE','SEX','DIAG13D','GEN1','GEN2','GEN3','GEN4','GEN5','GEN6'])
    # 25344 rows
    # get diagnosis '465'
    df1 = df.loc[df['DIAG13D'] != '000']
    df1 = df.loc[df['DIAG13D'] == '465']
    all_drugs = pd.concat([df1['GEN1'], df1['GEN2'], df1['GEN3'], df1['GEN4'],
                           df1['GEN5'], df1['GEN6']])
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

    df3 = pd.concat([df1, df2], axis=1)
    df3 = df3.reset_index(drop=True)
    df3.to_csv(filename.split('.')[0] + '_multidrug_AH.csv')


def find_freq(filename):
    df = pd.read_csv(filename)
    # scaling to unit variance
    df = df.drop(['VMONTH','AGE','Unnamed: 0', 'GEN1','GEN2','GEN3','GEN4','GEN5','GEN6','SEX','DIAG13D'], axis=1)
    freq = {}
    sum_elements = len(df)
    for column in df:
        freq[column] = sum(df[column])/sum_elements
    df2 = pd.DataFrame([freq], columns=freq.keys())


def classify(filename):
    df = pd.read_csv(filename)
    df['diag13d'] = df['diag13d'].astype(str)
    # scaling to unit variance
    df['normalAge']=StandardScaler().fit_transform(df['age'].reshape(-1,1))
    df['normalMonth'] = StandardScaler().fit_transform(df['vmonth'].reshape(-1, 1))
    df=df.drop(['age','vmonth','Index'], axis=1)
    x=df.ix[:,df.columns!='diag13d']
    y=df.ix[:, df.columns == 'diag13d']
    # training test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    lr_model = LogisticRegression(class_weight='balanced',multi_class='ovr')
    lr_model.fit(x_train, y_train)
    pred_test = lr_model.predict(x_test.values)
    print(pred_test)
    print(classification_report(y_test,pred_test))




def main():
    read_file('/Users/Amelia/Desktop/newfiles/ED00.csv')
    #find_freq('/Users/Amelia/Desktop/newfiles/ED00_multidrug_AH.csv')

if __name__ == '__main__':
        main()