import pandas as pd

df = pd.read_csv("data/dataset_54_vehicle.csv", sep=',')
minimums = [x[1].min() for x in df.iteritems()]
del minimums[-1]
maximums = [x[1].max() for x in df.iteritems()]
del maximums[-1]

newDf = pd.DataFrame()
k = 0
for column in df.columns.values.tolist()[:18]:
    newDf[column] = (df[column] - minimums[k]) / (maximums[k] - minimums[k])
    k = k + 1
newDf = newDf.join(df['Class'])
newDf = pd.get_dummies(newDf, columns=['Class'])
newDf.to_csv("data/normies.csv", index=False)
