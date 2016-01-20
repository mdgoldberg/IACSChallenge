import pandas as pd, numpy as np, matplotlib.pyplot as plt, utils, pickle
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

df1, df2 = utils.makeTrainCSVs('data/train.txt')
testdf = utils.makeTestCSV('data/test.txt')
origdf2 = df2.copy()

with open('data/neighbor_feat.pkl', 'rb') as f:
    neighbor_avgs = pickle.load(f)

neighbor_df = pd.DataFrame(neighbor_avgs,
                           columns=['S' + str(s) for s in xrange(1, 57)])

# cut = pd.cut(df2.minutes, xrange(-1, 1441, 30), labels=range(48))
# df2['bucket'] = cut.get_values()
newdf = df2.groupby(['day', 'weekday', 'hour', 'sensor', 'isRestroom', 'isStaircase']).value.sum().reset_index()
hours = (newdf['day'] - 1) * 24 + newdf['hour']
sensors = newdf['sensor'].str[1:].astype(int) - 1
newdf['neighbor_avg'] = [neighbor_df.ix[h, s] for h, s in zip(hours, sensors)]
newdf = utils.addSensorColumns(newdf)
dummies = pd.get_dummies(newdf.weekday, prefix='day').iloc[:, :-1]
newdf = pd.concat((newdf, dummies), axis=1)
comp_df = newdf.dropna()
Xdf = comp_df.ix[:, ['Xcoord', 'Ycoord', 'hour', 'neighbor_avg', 'isRestroom', 'isStaircase'] +
                 [c for c in comp_df.columns if c in dummies.columns]]
Y = comp_df.value.copy()

model = ensemble.GradientBoostingRegressor(n_estimators=200, loss='huber', max_depth=5)
grid = {'learning_rate': [0.1, 0.2, 0.25, 0.3, 0.35], 'alpha': [0.8, 0.85, 0.9, 0.95]}
model = GridSearchCV(model, grid, refit=True, n_jobs=-1, cv=4)
Xtrain, Xtest, ytrain, ytest = train_test_split(Xdf, Y, test_size=0.1)
print 'fitting...'
model.fit(Xtrain, ytrain)
cv_preds = model.predict(Xtest)
absErrs = np.abs(cv_preds - ytest).values
print np.mean(absErrs), np.median(absErrs)

testHours = (testdf['start_day'] - 1) * 24 + testdf['start_hour']
testSensors = testdf.sensor.str[1:].astype(int) - 1
testdf['neighbor_avg'] = [neighbor_df.ix[h, s] for h, s in zip(testHours, testSensors)]
newtest = testdf.ix[:, ['start_day', 'start_weekday', 'start_hour', 'Xcoord', 'Ycoord', 'neighbor_avg', 'isRestroom', 'isStaircase']]
dummies = pd.get_dummies(newtest.start_weekday, prefix='day').iloc[:, :-1]
newtest = pd.concat((newtest, dummies), axis=1)
comp_test = newtest.dropna()
Xtest = comp_test.ix[:, ['Xcoord', 'Ycoord', 'start_hour', 'neighbor_avg', 'isRestroom', 'isStaircase'] + 
                        [c for c in comp_test.columns if c.startswith('day_')]]

preds = model.predict(Xtest)
fi = pd.Series(dict(zip(Xtest.columns, model.best_estimator_.feature_importances_))).sort_values(ascending=True)
print fi
print model.best_estimator_
