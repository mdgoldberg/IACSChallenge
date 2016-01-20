import json
import numpy as np
import pandas as pd

def getDHM(raw_timestamp, index=['day', 'hour', 'minute']):
    day = raw_timestamp[:-4]
    hour = raw_timestamp[-4:-2]
    minute = raw_timestamp[-2:]
    return pd.Series([day, hour, minute], index=index)

def expand_rows(df):
    dfs = []
    for i in xrange(1, 57):
        temp_new_df = df.loc[:, [x for x in df.columns if x[0] != 'S']].copy()
        temp_new_df.loc[:, 'value'] = df['S' + str(i)]
        temp_new_df.loc[:, 'sensor'] = 'S' + str(i)
        dfs.append(temp_new_df)
        
    return (pd.concat(dfs)
            .sort_values(['day', 'minutes', 'sensor'])
            .reset_index(drop=True))

def addSensorColumns(df):
    coords = getCoords()
    df['Xcoord'] = df['sensor'].apply(lambda s: coords[s][0])
    df['Ycoord'] = df['sensor'].apply(lambda s: coords[s][1])
    df['isRestroom'] = df['sensor'].isin(['S6', 'S9', 'S12', 'S44'])
    df['isStaircase'] = df['sensor'].isin(['S35', 'S42', 'S52'])
    return df

def addDummyRows(df, hourCol):
    temp = pd.DataFrame(index=xrange(24*56), columns=[hourCol, 'sensor', 'isDummy'])
    for h in xrange(24):
        for s in xrange(1, 57):
            idx = 56*h + s - 1
            temp.ix[idx, hourCol] = h
            temp.ix[idx, 'sensor'] = 'S' + str(s)
    temp['isDummy'] = True
    temp['isRestroom'] = True
    df = df.append(temp).reset_index(drop=True)
    df.ix[df.isDummy.isnull(), 'isDummy'] = False
    df.isDummy = df.isDummy.astype(bool)
    return df

def makeTrainCSVs(fn):
    df = pd.read_csv(fn)
    df = df.replace(-1, np.nan)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={'Timestamp (DHHMM)': 'raw_timestamp'}, inplace=True)
    times = df.raw_timestamp.astype(str).apply(getDHM).astype(int)
    df = pd.concat((df, times), axis=1)
    df['minutes'] = df['hour'] * 60 + df['minute']
    df['hours'] = (df['day'] - 1) * 24 + df['hour']
    df['weekday'] = df['day'] % 7
    df2 = expand_rows(df)
    df2 = addSensorColumns(df2)
    df2 = addDummyRows(df2, 'hour')
    return df, df2

def makeTestCSV(fn):
    df = pd.read_csv(fn)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={'Sensor ID': 'sensor', 'Start Time': 'start_time',
                       'End Time': 'end_time'}, inplace=True)
    df.drop('People Count', axis=1, inplace=True)
    start = (df.start_time.astype(str).apply(
        lambda d: getDHM(d, ['start_day', 'start_hour', 'start_minute'])
    ).astype(int))
    end = (df.end_time.astype(str).apply(
        lambda d: getDHM(d, ['end_day', 'end_hour', 'end_minute'])
    ).astype(int))
    df = pd.concat((df, start, end), axis=1)
    df['start_minutes'] = df['start_hour'] * 60 + df['start_minute']
    df['start_hours'] = (df['start_day'] - 1) * 24 + df['start_hour']
    df['end_minutes'] = df['end_hour'] * 60 + df['end_minute']
    df['end_hours'] = (df['end_day'] - 1) * 24 + df['end_hour']
    df['start_weekday'] = df['start_day'] % 7
    df['end_weekday'] = df['end_day'] % 7
    df = addSensorColumns(df)
    df['isRestroom'] = df['sensor'].isin(['S6', 'S9', 'S12', 'S44'])
    df['isStaircase'] = df['sensor'].isin(['S35', 'S42', 'S52'])
    df = addDummyRows(df, 'start_hour')
    return df

def manhattan(t1, t2):
    return abs(t1[0] - t2[0]) + abs(t1[1] - t2[1])

def euclidean(t1, t2):
    return np.sqrt((t1[0] - t2[0]) ** 2. + (t1[1] - t2[1]) ** 2.)

def getCoords():
    with open('data-final/coord_dict.json') as f:
        coords = json.load(f)
    return coords

def find_k_nearest_all(k):
    coords = getCoords()
    def find_k_nearest_sensor(sensor, k):
        out = np.array([manhattan(coords['S' + str(sensor)], 
                                  coords['S' + str(i)]) for i in xrange(1, 56)]).argsort()[1:k + 1] + 1
        return map(lambda x: 'S' + str(x), out)    
    neighbors = {}
    for i in xrange(1, 57):
        neighbors['S' + str(i)] = find_k_nearest_sensor(i, k)
    return neighbors

def makeKaggleSubmission(preds, fn):
    with open(fn, 'w') as f:
        f.write('Index,Count\n')
        for i, p in enumerate(preds):
            f.write('{},{}\n'.format(i+1, p))

def makeFinalSubmission(preds, fn):
    with open(fn, 'w') as f:
        for p in preds:
            f.write('{}\n'.format(int(np.around(p))))
