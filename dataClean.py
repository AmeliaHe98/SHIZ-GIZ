import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# read three type of files and extract key info
# combine three into one matrix
ED02SPS_labels = []


def read_ed02sps(filename):
    # rfv'n' = "Patient's Reason for Visit 'n'"
    # diag13d = "Physician's diagnosis #1 - broad" - main diagnosis
    # 37337 rows
    df = pd.read_csv(filename, usecols=['vmonth', 'age','sex','rfv1','rfv2','rfv3','diag13d'])
    # 36603 rows
    df1 = df.loc[df['diag13d']!='000']
    # add new col to indicate if diag is within top 35
    df1['top35'] = 0
    df2 = df1
    df2['rfv1'] = 0
    df2['rfv2'] = 0
    df2['rfv3'] = 0
    dicts = df2['diag13d'].value_counts()
    dict(dicts)
    sorted_dicts = sorted(dicts.iteritems(), key=lambda x: x[1], reverse=True)
    labels = []
    for each in range(35):
        labels.append(sorted_dicts[each][0])
    global ED02SPS_labels
    ED02SPS_labels = labels
    df3 = df2.loc[df2['diag13d'].isin(labels)]
    df3 = df3.reset_index(drop=True)
    df3.to_csv(filename.split('.')[0]+'_35_AH.csv')


def read_ed02sps1(filename):
    df = pd.read_csv(filename)
    count_classes = pd.value_counts(df['diag13d'])
    count_classes.plot(kind='bar')
    plt.show()


def two_case(filename, labels):
    df = pd.read_csv(filename)
    df = df.drop(columns=['Unnamed: 0'])
    df1 = []
    df2 = []
    for each in labels:
        df1 = df.loc[df['diag13d'] == each]
        df2 = df.loc[df['diag13d'] != each]
    df1['top35'] = 1
    df2['top35'] = 0
    df3 = pd.concat([df1, df2], ignore_index=True)
    df3.to_csv(filename.split('.')[0] + '_two_case.csv', index_label='Index')


def main():
    read_ed02sps('/Users/Amelia/Desktop/model/ED02SPS.csv')
    read_ed02sps1('/Users/Amelia/Desktop/model/ED02SPS_35_AH.csv')
    two_case('/Users/Amelia/Desktop/model/ED02SPS_35_AH.csv', ED02SPS_labels)


if __name__ == '__main__':
    main()