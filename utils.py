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
        
    return pd.concat(dfs).sort_values(['day', 'minutes', 'sensor'])

def makeTrainCSVs(fn):
    df = pd.read_csv(fn)
    df = df.replace(-1, np.nan)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={'Timestamp (DHHMM)': 'raw_timestamp'}, inplace=True)
    times = df.raw_timestamp.astype(str).apply(getDHM).astype(int)
    df = pd.concat((df, times), axis=1)
    df['minutes'] = df['hour'] * 60 + df['minute']
    df['weekday'] = df['day'] % 7
    return df, expand_rows(df)

def makeTestCSV(fn):
    pass

def manhattan(t1, t2):
    return abs(t1[0] - t2[0]) + abs(t1[1] - t2[1])

def euclidean(t1, t2):
    return np.sqrt((t1[0] - t2[0]) ** 2. + (t1[1] - t2[1]) ** 2.)

def find_k_nearest_all(k):
    coords = json.load(file('data/coord_dict.json'))
    def find_k_nearest_sensor(sensor, k):
        out = np.array([utils.manhattan(coords['S' + str(sensor)], 
                                        coords['S' + str(i)]) for i in xrange(1, 56)]).argsort()[1:k + 1] + 1
        return map(lambda x: 'S' + str(x), out)    
    neighbors = {}
    for i in xrange(1, 57):
        neighbors['S' + str(i)] = find_k_nearest_sensor(i, k)
    return neighbors
