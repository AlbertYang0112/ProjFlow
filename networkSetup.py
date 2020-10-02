import numpy as np
import pandas as pd
import networkx as nx
import datetime as dt
from tqdm import tqdm
import matplotlib.pyplot as plt

AUSTIN_TIME_OFFSET = -6 * 60 * 60
DATA_TIME_OFFSET = 0
DAY_LENGTH = 24 * 60 * 60

def geohashMix(x, y):
    return ((x << 16) | y) & 0xFFFFFFFF

def geohashSep(c):
    return c >> 16, c & 0xFFFF

def getAustinTime(timeStamp):
    return dt.datetime.fromtimestamp(timeStamp, tz=dt.timezone(dt.timedelta(0, DATA_TIME_OFFSET)))

data = pd.read_csv("proc/01.csv")
baseUnixStamp = (data.loc[0, 'location_at'] + DATA_TIME_OFFSET) // DAY_LENGTH * DAY_LENGTH - DATA_TIME_OFFSET
graphInterval = 60 * 60
print(np.min(data.loc[:, ('bx', 'by')], axis=0), np.max(data.loc[:, ('bx', 'by')], axis=0))

def netSetup(data):
    G = nx.DiGraph()
    nodes = np.unique(data.loc[:, ("bx", "by")], axis=0)
    G.add_nodes_from([(geohashMix(x, y), {'viz': {'position': {'x': x, 'y': y}}}) for x, y in nodes])
    publisherList = np.unique(data['publisher_id'])
    for publisher in tqdm(publisherList):
        path = data.loc[data[data['publisher_id'] == publisher].index].sort_values(by='location_at').loc[:, ('bx', 'by')].values
        for i in range(len(path) - 1):
            u = geohashMix(path[i][0], path[i][1])
            v = geohashMix(path[i + 1][0], path[i + 1][1])
            if (u, v) in G.edges():
                G.edges[u, v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)
    return G

