import pandas as pd, numpy as np, matplotlib.pyplot as plt, utils
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

df1, df2 = utils.makeTrainCSVs('data/train.txt')
testdf = utils.makeTestCSV('data/test.txt')
coords = utils.getCoords()
origdf2 = df2.copy()

cut = pd.cut(df2.minutes, xrange(-1, 1441, 30), labels=range(48))
df2['bucket'] = cut.get_values()
newdf = df2.groupby(['day', 'weekday', 'bucket', 'sensor']).value.sum().reset_index()
dummies = pd.get_dummies(newdf.weekday, prefix='day').iloc[:, :-1]
newdf['Xcoord'] = newdf.sensor.apply(lambda s: coords[s][0])
newdf['Ycoord'] = newdf.sensor.apply(lambda s: coords[s][1])
newdf = pd.concat((newdf, dummies), axis=1)
comp_df = newdf.dropna()
Xdf = comp_df.ix[:, ['Xcoord', 'Ycoord', 'bucket'] +
                 [c for c in comp_df.columns if c.startswith('day_')]]
Y = comp_df.value.copy()

rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
Xtrain, Xtest, ytrain, ytest = train_test_split(Xdf, Y, test_size=0.1)
rfr.fit(Xtrain, ytrain)
preds = rfr.predict(Xtest)
absErrs = np.abs(preds - ytest).values
print np.mean(absErrs), np.median(absErrs)

testdf
