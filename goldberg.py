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
hours = map(int, (newdf['day'] - 1) * 24 + newdf['hour'])
sensors = newdf['sensor'].str[1:].astype(int) - 1
newdf['neighbor_avg'] = [neighbor_df.ix[h, s] for h, s in zip(hours, sensors)]
newdf = utils.addSensorColumns(newdf)
newdf = utils.addDummyRows(newdf, 'hour')
day_dummies = pd.get_dummies(newdf.weekday, prefix='day').iloc[:, :-1]
sens_dummies = pd.get_dummies(newdf.sensor).iloc[:, :-1]
newdf = pd.concat((newdf, day_dummies, sens_dummies), axis=1)
comp_df = newdf.dropna()
Xdf = comp_df.ix[:, ['Xcoord', 'Ycoord', 'hour', 'neighbor_avg', 'isRestroom', 'isStaircase'] +
                 [c for c in comp_df.columns if c in day_dummies.columns or c in sens_dummies.columns]]
Y = comp_df.value.copy()

# model = ensemble.RandomForestRegressor(n_estimators=200, n_jobs=-1)
# grid = {'max_features': ['auto', 'sqrt', 'log2', 0.25, 0.5, 0.75]}
# model = GridSearchCV(model, grid, refit=True, n_jobs=-1, verbose=2)
model = ensemble.GradientBoostingRegressor(n_estimators=200, loss='huber', max_depth=5)
grid = {'learning_rate': [0.05, 0.1, 0.15], 'alpha': [0.9, 0.95, 0.99]}
model = GridSearchCV(model, grid, refit=True, n_jobs=-1, verbose=2)
Xtrain, Xtest, ytrain, ytest = train_test_split(Xdf, Y, test_size=0.1)
print 'fitting...'
model.fit(Xtrain, ytrain)
cv_preds = model.predict(Xtest)
absErrs = np.abs(cv_preds - ytest).values
print np.mean(absErrs), np.median(absErrs)

day_dummies = pd.get_dummies(testdf.start_weekday, prefix='day').iloc[:, :-1]
sens_dummies = pd.get_dummies(testdf.sensor).iloc[:, :-1]
newtest = pd.concat((testdf, day_dummies, sens_dummies), axis=1)
newtest = newtest.ix[~newtest.isDummy]
testHours = map(int, (newtest['start_day'] - 1) * 24 + newtest['start_hour'])
testSensors = newtest.sensor.str[1:].astype(int) - 1
newtest['neighbor_avg'] = [neighbor_df.ix[h, s] for h, s in zip(testHours, testSensors)]
comp_test = newtest.dropna()
Xtest = comp_test.ix[:, ['Xcoord', 'Ycoord', 'start_hour', 'neighbor_avg', 'isRestroom', 'isStaircase'] + 
                        [c for c in comp_test.columns if c in day_dummies.columns or c in sens_dummies.columns]]

preds = model.predict(Xtest)
preds = [p if p >= 0 else 0 for p in preds]
fi = pd.Series(dict(zip(Xtest.columns, model.best_estimator_.feature_importances_))).sort_values(ascending=True)
print fi
print model.best_estimator_
