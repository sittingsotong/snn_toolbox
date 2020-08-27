import pandas as pd

path='../temp/2bits_2/log/gui/test/accuracy.txt'

data = pd.read_csv(path, sep=" ", header=None, skiprows=[0])

data.columns = ["index", "snn top 1", "top 1", "ann top 1", "ann"]

data['snn top 1'] = data['snn top 1'].str[:-1].astype(float)
data['top 1'] = data['top 1'].str[:-1].astype(float)
data['ann top 1'] = data['ann top 1'].str[:-1].astype(float)
data['ann'] = data['ann'].str[:-1].astype(float)

sum = data.sum(axis = 0, skipna = True)
result = sum/30

print(result)