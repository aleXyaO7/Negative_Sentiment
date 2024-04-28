import pandas as pd

df = pd.read_csv('original_dataset.csv', header=None).iloc[:, :2]

positive = set([0, 1, 4, 5, 6, 7, 8, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 27])
negative = set([2, 3, 9, 10, 11, 12, 14, 16, 24, 25])

for i in range(len(df.iloc[:, 1])):
    temp = df[1][i].split(',')
    flag = False
    for j in temp:
        if int(j) in negative:
            df[1][i] = 1
            flag = True
    if not flag:
        df[1][i] = 0

df.columns = ['sentence', 'label']

df.to_csv('dataset.csv', index=None)