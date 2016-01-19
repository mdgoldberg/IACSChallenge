import numpy as np
import pandas as pd

def getDHM(raw_timestamp, index=['day', 'hour', 'minute']):
    day = raw_timestamp[:-4]
    hour = raw_timestamp[-4:-2]
    minute = raw_timestamp[-2:]
    return pd.Series([day, hour, minute], index=index)
