import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
# read three type of files and extract key info
# combine three into one matrix
top_labels = []


def read_file(filename, n):
    # rfv'n' = "Patient's Reason for Visit 'n'"
    # diag13d = "Physician's diagnosis #1 - broad" - main diagnosis
    # 37337 rows
    df = pd.read_csv(filename, usecols=['vmonth', 'age','sex','rfv1','rfv2','rfv3','diag13d'])
    # 36603 rows
    df1 = df.loc[df['diag13d']!='000']
    # add new col to indicate if diag is within top 35
    df1['topn'] = 0
    df2 = pd.get_dummies(df1,columns=['rfv1','rfv2','rfv3'])
    dicts = df2['diag13d'].value_counts()
    dict(dicts)
    sorted_dicts = sorted(dicts.iteritems(), key=lambda x: x[1], reverse=True)
    labels = []
    for each in range(n):
        labels.append(sorted_dicts[each][0])

    global top_labels
    top_labels = labels
    df3 = df2.loc[df2['diag13d'].isin(labels)]
    df3 = df3.reset_index(drop=True)
    df3.to_csv(filename.split('.')[0]+'_multi_AH.csv')


def read_file1(filename):
    df = pd.read_csv(filename)
    count_classes = pd.value_counts(df['diag13d'])
    count_classes.plot(kind='bar')
    plt.show()


def two_case(filename, labels):
    df = pd.read_csv(filename)
    df = df.drop(columns=['Unnamed: 0'])
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    for each in labels:
        df1 = pd.concat([df1, df.loc[df['diag13d'] == each]])
        df2 = pd.concat([df1, df.loc[df['diag13d'] != each]])
    df1['topn'] = 1
    df2['topn'] = 0
    df3 = pd.concat([df1, df2], ignore_index=True)
    df3.to_csv(filename.split('.')[0] + '_two_case.csv', index_label='Index')


def combine_files(f1, f2, f3):
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    df3 = pd.read_csv(f3)
    frames = [df1, df2, df3]
    result = pd.concat(frames)
    result.to_csv('/Users/Amelia/Desktop/model/combined_files.csv', index_label='Index')


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
    # read_file('/Users/Amelia/Desktop/model/ED02SPS.csv', 35)
    # read_file1('/Users/Amelia/Desktop/model/ED02SPS_multi_AH.csv')
    # two_case('/Users/Amelia/Desktop/model/ED02SPS_multi_AH.csv', top_labels)
    # read_file('/Users/Amelia/Desktop/model/NAM03SPS.csv', 35)
    # read_file1('/Users/Amelia/Desktop/model/NAM03SPS_multi_AH.csv')
    # two_case('/Users/Amelia/Desktop/model/NAM03SPS_multi_AH.csv', top_labels)
    # read_file('/Users/Amelia/Desktop/model/OPD02SPS.csv', 35)
    # read_file1('/Users/Amelia/Desktop/model/OPD02SPS_multi_AH.csv')
    # two_case('/Users/Amelia/Desktop/model/OPD02SPS_multi_AH.csv', top_labels)
    # combine_files('/Users/Amelia/Desktop/model/ED02SPS_multi_AH_two_case.csv','/Users/Amelia/Desktop/model/NAM03SPS_multi_AH_two_case.csv',
#                 '/Users/Amelia/Desktop/model/OPD02SPS_multi_AH_two_case.csv')
    classify('/Users/Amelia/Desktop/model/OPD02SPS_multi_AH_two_case.csv')


if __name__ == '__main__':
    main()