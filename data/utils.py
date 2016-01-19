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
        temp_new_df = df[[x for x in df.columns if x[0] != 'S']]
        temp_new_df['value'] = df['S' + str(i)]
        temp_new_df['sensor'] = 'S' + str(i)
        dfs.append(temp_new_df)
        
    return pd.concat(dfs)

def makeTrainCSVS(fn):
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

