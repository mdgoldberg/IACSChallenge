import pandas as pd, numpy as np, matplotlib.pyplot as plt, utils, pickle
from sklearn import ensemble
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
# closestSensors = newdf.sensor.apply(closeDict.get).to_dict()
newdf = utils.addSensorColumns(newdf)
newdf = utils.addDummyRows(newdf, 'hour')
day_dummies = pd.get_dummies(newdf.weekday, prefix='day').iloc[:, :-1]
sens_dummies = pd.get_dummies(newdf.sensor).iloc[:, :-1]
newdf = pd.concat((newdf, day_dummies, sens_dummies), axis=1)
comp_df = newdf.dropna()
Xdf = comp_df.ix[:, ['Xcoord', 'Ycoord', 'hour', 'neighbor_avg', 'total_mean', 'isRestroom', 'isStaircase'] +
                 list(day_dummies.columns)
                 # + list(sens_dummies.columns)
                 ]
Y = comp_df.value.copy()

# model = ensemble.RandomForestRegressor(n_jobs=-1)
# grid = {'n_estimators': [100, 150, 200], 'max_features': ['auto', 'sqrt', 'log2']}
model = ensemble.GradientBoostingRegressor(loss='huber', n_estimators=200, max_depth=5)
grid = {'learning_rate': [0.05, 0.1, 0.15], 'alpha': [0.99, 0.999, 0.9999]}
model = GridSearchCV(model, grid, refit=True, n_jobs=-1, verbose=2)
Xtrain, Xtest, ytrain, ytest = train_test_split(Xdf, Y, test_size=0.1)
print 'fitting...'
model.fit(Xtrain, ytrain)
cv_preds = model.predict(Xtest)
absErrs = np.abs(cv_preds - ytest).values
print np.mean(absErrs), np.median(absErrs)
# model.fit(Xdf, Y)

day_dummies = pd.get_dummies(testdf.start_weekday, prefix='day').iloc[:, :-1]
sens_dummies = pd.get_dummies(testdf.sensor).iloc[:, :-1]
newtest = pd.concat((testdf, day_dummies, sens_dummies), axis=1)
newtest = newtest.ix[~newtest.isDummy]
testHours = newtest['start_hours'].astype(int)
testSensors = newtest.sensor.str[1:].astype(int) - 1
newtest['neighbor_avg'] = [neighbor_df.ix[h, s] for h, s in zip(testHours, testSensors)]
newtest['total_mean'] = newtest.start_hours.apply(hoursMeansDict.get)
comp_test = newtest.dropna()
Xtest = comp_test.ix[:, ['Xcoord', 'Ycoord', 'start_hour', 'neighbor_avg', 'total_mean', 'isRestroom', 'isStaircase'] + 
                     list(day_dummies.columns)
                     # + list(sens_dummies.columns)
                     ]

preds = model.predict(Xtest)
preds = [p if p >= 0 else 0 for p in preds]
fi = pd.Series(dict(zip(Xtest.columns, model.best_estimator_.feature_importances_))).sort_values(ascending=True)
print fi
print model.best_estimator_
