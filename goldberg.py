import pandas as pd, numpy as np, matplotlib.pyplot as plt, utils, pickle, sys
from sklearn import ensemble
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

df1, df2 = utils.makeTrainCSVs('data/train.txt')
testdf = utils.makeTestCSV('data/test.txt')
origdf2 = df2.copy()

with open('data/neighbor_feat.pkl', 'rb') as f:
    neighbor_avgs = pickle.load(f)

closeDict = utils.find_k_nearest_all(56)

neighbor_df = pd.DataFrame(neighbor_avgs,
                           columns=['S' + str(s) for s in xrange(1, 57)])

# cut = pd.cut(df2.minutes, xrange(-1, 1441, 30), labels=range(48))
# df2['bucket'] = cut.get_values()
newdf = df2.groupby(['day', 'weekday', 'hour', 'hours', 'sensor', 'isRestroom', 'isStaircase']).value.sum().reset_index()
hours = newdf['hours'].astype(int)
sensors = newdf['sensor'].str[1:].astype(int) - 1
newdf['neighbor_avg'] = [neighbor_df.ix[h, s] for h, s in zip(hours, sensors)]
hoursMeansDict = newdf.groupby('hours').value.mean().to_dict()
newdf['total_mean'] = newdf.hours.apply(hoursMeansDict.get)
hsDict = newdf.groupby(['hours', 'sensor']).value.mean().to_dict()
closestSensors = newdf.sensor.apply(closeDict.get).values
topdf = pd.DataFrame([pd.Series([hsDict[h, s] for s in l]).dropna().values[:30] for h, l in zip(hours, closestSensors)])
topdf.columns = ['top_{}'.format(i) for i in xrange(30)]
newdf = utils.addSensorColumns(newdf)
newdf = utils.addDummyRows(newdf, 'hour')
day_dummies = pd.get_dummies(newdf.weekday, prefix='day').iloc[:, :-1]
sens_dummies = pd.get_dummies(newdf.sensor).iloc[:, :-1]
sens_dummies = np.multiply(sens_dummies, newdf.hours.apply(hoursMeansDict.get)[:, np.newaxis])
newdf = pd.concat((newdf, day_dummies, sens_dummies, topdf), axis=1)
comp_df = newdf.dropna()
Xdf = comp_df.ix[:, ['Xcoord', 'Ycoord', 'hour', 'neighbor_avg', 'total_mean', 'isRestroom', 'isStaircase'] +
                 list(day_dummies.columns)
                 + list(sens_dummies.columns)
                 + list(topdf.columns)
                 ]
Y = comp_df.value.copy()
nf = Xdf.shape[1]

# model = ensemble.RandomForestRegressor(n_jobs=-1)
# grid = {'n_estimators': [100, 150, 200], 'max_features': ['auto', 'sqrt', 'log2']}
# model = ensemble.GradientBoostingRegressor(loss='huber', n_estimators=200, max_depth=5)
# grid = {'learning_rate': [0.05, 0.1, 0.15], 'alpha': [0.95, 0.99, 0.999]}
model = SVR()
grid = {'C': [1e-2, 1e-1, 0.25], 'epsilon': [1e-2, 1e-1, 0.2], 'gamma': [0.5/nf, 1./nf, 2./nf]}
model = GridSearchCV(model, grid, refit=True, n_jobs=-1, verbose=2, cv=2)
Xtrain, Xtest, ytrain, ytest = train_test_split(Xdf, Y, test_size=0.1)
print 'fitting...'
model.fit(Xtrain, ytrain)
cv_preds = model.predict(Xtest)
absErrs = np.abs(cv_preds - ytest).values
print np.mean(absErrs), np.median(absErrs)
# model.fit(Xdf, Y)

day_dummies = pd.get_dummies(testdf.start_weekday, prefix='day').iloc[:, :-1]
sens_dummies = pd.get_dummies(testdf.sensor).iloc[:, :-1]
newtest = pd.concat((testdf, day_dummies), axis=1)
good_idx = ~newtest.isDummy
newtest = newtest.ix[~newtest.isDummy]
testHours = newtest['start_hours'].astype(int)
testSensors = newtest.sensor.str[1:].astype(int) - 1
newtest['neighbor_avg'] = [neighbor_df.ix[h, s] for h, s in zip(testHours, testSensors)]
newtest['total_mean'] = newtest.start_hours.apply(hoursMeansDict.get)
sens_dummies = sens_dummies.ix[good_idx]
sens_dummies = np.multiply(sens_dummies, newtest.start_hours.apply(hoursMeansDict.get)[:, np.newaxis])
newtest = pd.concat((newtest, sens_dummies), axis=1)

closestSensors = newtest.sensor.apply(closeDict.get).values
topdf = pd.DataFrame([pd.Series([hsDict[h, s] for s in l]).dropna().values[:30] for h, l in zip(testHours, closestSensors)])
topdf.columns = ['top_{}'.format(i) for i in xrange(30)]
newtest = pd.concat((newtest, topdf), axis=1)
comp_test = newtest.dropna()
Xtest = comp_test.ix[:, ['Xcoord', 'Ycoord', 'start_hour', 'neighbor_avg', 'total_mean', 'isRestroom', 'isStaircase'] + 
                     list(day_dummies.columns)
                     + list(sens_dummies.columns)
                     + list(topdf.columns)
                     ]

preds = model.predict(Xtest)
preds = [p if p >= 0 else 0 for p in preds]
fi = pd.Series(dict(zip(Xtest.columns, model.best_estimator_.feature_importances_))).sort_values(ascending=True)
print fi
print model.best_estimator_
