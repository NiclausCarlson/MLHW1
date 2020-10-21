import pandas as pd

df = pd.read_csv("data/dataset_191_wine.csv", sep=',')
column = df["class"]
df = df.drop("class", axis=1)
df["class"] = column

minimums = [x[1].min() for x in df.iteritems()]
del minimums[-1]
maximums = [x[1].max() for x in df.iteritems()]
del maximums[-1]

newDf = pd.DataFrame()
k = 0
for column in df.columns.values.tolist()[:13]:
    newDf[column] = (df[column] - minimums[k]) / (maximums[k] - minimums[k])
    k = k + 1
newDf = newDf.join(df['class'])
newDf = pd.get_dummies(newDf, columns=['class'])
newDf.to_csv("data/normies.csv", index=False)
