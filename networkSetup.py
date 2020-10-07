import os
from platform import node
import sys
import numpy as np
import pandas as pd
import networkx as nx
import datetime as dt
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

AUSTIN_TIME_OFFSET = -6 * 60 * 60
DATA_TIME_OFFSET = 0
DAY_LENGTH = 24 * 60 * 60
DEFAULT_TIME_INTERVAL = 180

parser = argparse.ArgumentParser(description="Generate The Map")
parser.add_argument('inputDir', type=str, metavar="DirIn", help="Preprocessed Dataset")
parser.add_argument('adjFile', type=str, metavar="AdjFile", help='Adjacency Matrix File')
parser.add_argument('featureFile', type=str, metavar="FeatureFile", help='Feature File')
parser.add_argument("--interval", type=int, metavar='interval', help='time interval (minute)', default=DEFAULT_TIME_INTERVAL)
parser.add_argument('--edgeTh', type=int, metavar="nodeTh", help="Edge Filtering Threshold", default=2)
parser.add_argument('--sigma', type=float, metavar="Sigma", help="Edge Weight Scaling Factor", default=1)
parser.add_argument('--pingTh', type=int, metavar="pingTh", help="Edge Filtering Threshold", default=10)
parser.add_argument('--nxFileDir', type=str, metavar="nxFileDir", help="Path to dump the networkx file.", default='')
parser.add_argument('--adjFig', type=str, metavar="adjFig", help="Path to save the figure of adjacency matrix.", default='')

def geohashMix(x, y):
    return ((x << 16) | y) & 0xFFFFFFFF

def geohashSep(c):
    return c >> 16, c & 0xFFFF

def getAustinTime(timeStamp):
    return dt.datetime.fromtimestamp(timeStamp, tz=dt.timezone(dt.timedelta(0, DATA_TIME_OFFSET)))

def getTimeStr(time):
    return f"{time.year}-{time.month}-{time.day}-{time.hour}-{time.minute}"

def netSetupWithPath(data):
    G = nx.DiGraph()
    nodes = np.unique(data.loc[:, ("bx", "by")], axis=0)
    G.add_nodes_from([(geohashMix(x, y), {'viz': {'position': {'x': x, 'y': y}}}) for x, y in nodes])
    advertiserList = np.unique(data['advertiser_id'])
    for advertiser in tqdm(advertiserList):
        path = data.loc[data[data['advertiser_id'] == advertiser].index].sort_values(by='location_at').loc[:, ('bx', 'by')].values
        for i in range(len(path) - 1):
            u = geohashMix(path[i][0], path[i][1])
            v = geohashMix(path[i + 1][0], path[i + 1][1])
            if (u, v) in G.edges():
                G.edges[u, v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)
    return G

def netSetupWithCount(data, distTh, countTh):
    # Todo: Many redundant computations. Remove them
    G = nx.Graph()
    nodes, nodeCounts = np.unique(data.loc[:, ("bx", "by")], axis=0, return_counts=True)
    validNodes = nodes[nodeCounts > countTh]
    validCounts = nodeCounts[nodeCounts > countTh]
    G.add_nodes_from([
        (geohashMix(x, y), {'weight': c, 'viz': {'position': {'x': x, 'y': y, 'z': 0}}}) for (x, y), c in zip(validNodes, validCounts)
    ])
    nodeHash = np.bitwise_or(np.left_shift(validNodes[:, 0], 16), validNodes[:, 1])
    nodeHashSet = set(nodeHash)
    for i, (x, y) in enumerate(validNodes):
        meshLb = np.max([[0, x-distTh], [0, y-distTh]], axis=1)
        meshUb = np.min([[0xFFFF, x+distTh], [0xFFFF, y+distTh]], axis=1)
        deltaX = np.linspace(meshLb[0], meshUb[0], meshUb[0] - meshLb[0] + 1, dtype=np.int)
        deltaY = np.linspace(meshLb[1], meshUb[1], meshUb[1] - meshLb[1] + 1, dtype=np.int)
        meshX, meshY = np.meshgrid(deltaX, deltaY)
        hashCode = np.bitwise_or(np.left_shift(meshX, 16), meshY)
        for code in hashCode.flatten():
            if code in nodeHashSet and (code, nodeHash[i]) not in G.edges():
                tx, ty = geohashSep(code)
                dist = np.sqrt((tx - x) ** 2 + (ty - y) ** 2)
                if dist < 1:
                    dist = 1
                G.add_edge(code, nodeHash[i], weight = 1.0 / dist)
    return G

def dataSplit(data, interval):
    baseUnixStamp = (data.loc[0, 'location_at'] + DATA_TIME_OFFSET) // DAY_LENGTH * DAY_LENGTH - DATA_TIME_OFFSET
    splitCnt = int(DAY_LENGTH_MIN / interval)
    idxList = []
    for splitIdx in range(splitCnt):
        timeStampLb = splitIdx * interval * 60 + baseUnixStamp
        timeStampUb = (splitIdx + 1) * interval * 60 + baseUnixStamp
        sel = np.logical_and(data['location_at'] >= timeStampLb, data['location_at'] < timeStampUb)
        idx = data[sel].index
        idxList.append((idx, getAustinTime(timeStampLb)))
    return idxList

if __name__ == '__main__':
    args = parser.parse_args()
    interval = args.interval
    DAY_LENGTH_MIN = 24 * 60
    if DAY_LENGTH_MIN % interval != 0:
        print(f"Day cannot be splitted into intervals with length of {interval} min.")
        interval = DAY_LENGTH_MIN / int(DAY_LENGTH_MIN / interval)
        print(f"Using {interval} instead")
    # Find all the CSV files
    csvFiles = []
    for path, _, files in os.walk(args.inputDir):
        for file in files:
            if file[-4:] == '.csv' and file != 'stat.csv':
                csvFiles.append((path, file))
    print(f"Found {len(csvFiles)} files under {args.inputDir}")
    fileLoader = tqdm(csvFiles)
    nodeList = []
    nodeCountList = []
    nodeTimeList = []
    if len(args.nxFileDir) > 0:
        os.makedirs(args.nxFileDir, exist_ok=True)
    # Collect all the nodes and ping counts
    for p, f in fileLoader:
        filePath = os.path.join(p, f)
        rawData = pd.read_csv(filePath)
        dataIdx = dataSplit(rawData, interval)
        for idx, t in dataIdx:
            dataSlice = rawData.loc[idx]
            nodes, nodeCount = np.unique(dataSlice.loc[:, ("bx", "by")], axis=0, return_counts=True)
            validNodeIdx = nodeCount > args.pingTh
            validNodes = nodes[validNodeIdx]
            validNodeCounts = nodeCount[validNodeIdx]
            nodeList.append(validNodes)
            nodeCountList.append(validNodeCounts)
            nodeTimeList.append(t)
            if len(args.nxFileDir) > 0:
                graphSlice = netSetupWithCount(dataSlice, args.edgeTh, args.pingTh)
                nx.write_gexf(graphSlice, os.path.join(args.nxFileDir, getTimeStr(t)+'.gexf'))
    
    # Setup the global graph
    ## Collect the nodes
    featureCnt = int(len(csvFiles) * (DAY_LENGTH_MIN / interval))
    assert featureCnt == len(nodeTimeList)
    node = np.unique(np.concatenate(nodeList, axis=0), axis=0)
    nodeHash = np.bitwise_or(np.left_shift(node[:, 0], 16), node[:, 1])
    order = np.argsort(nodeHash)
    node = node[order]
    nodeHash = nodeHash[order]
    nodeCnt = int(node.shape[0])
    print(f"Found {nodeCnt} unique nodes")
    ## Collect the features
    counts = np.zeros((nodeCnt, featureCnt), dtype=np.int)
    timeOrder = np.argsort(nodeTimeList)
    print(f"Collect the features")
    for i in tqdm(range(featureCnt), total=featureCnt):
        featureIdx = timeOrder[i]
        for n, c in zip(nodeList[featureIdx], nodeCountList[featureIdx]):
            idx = np.where(np.all(node == n, axis=1))
            assert len(idx) == 1, f"Node: {n}, where: {idx}"
            idx = idx[0]
            counts[idx, featureIdx] += c
    # Construct the Graph
    print("Construct the Graph")
    G = nx.Graph()
    nodeWeight = np.sum(counts, axis=1)
    nodeHashSet = set(nodeHash)
    for idx, (x, y) in tqdm(enumerate(node)):
        G.add_node(nodeHash[idx], weight=nodeWeight[idx], viz={'position': {'x': x, 'y': y, 'z': 0}})
        meshLb = np.max([[0, x-args.edgeTh], [0, y-args.edgeTh]], axis=1)
        meshUb = np.min([[0xFFFF, x+args.edgeTh], [0xFFFF, y+args.edgeTh]], axis=1)
        deltaX = np.linspace(meshLb[0], meshUb[0], meshUb[0] - meshLb[0] + 1, dtype=np.int)
        deltaY = np.linspace(meshLb[1], meshUb[1], meshUb[1] - meshLb[1] + 1, dtype=np.int)
        meshX, meshY = np.meshgrid(deltaX, deltaY)
        meshHashCode = np.bitwise_or(np.left_shift(meshX, 16), meshY)
        for code in meshHashCode.flatten():
            if code in nodeHashSet and \
                code != nodeHash[idx] and \
                (code, nodeHash[idx]) not in G.edges():
                tx, ty = geohashSep(code)
                dist2 = (tx - x) ** 2 + (ty - y) ** 2
                dist = np.sqrt(dist2)
                if dist > args.edgeTh:
                    continue
                edgeWeight = np.exp(-dist2 / (args.sigma ** 2))
                G.add_edge(code, nodeHash[idx], weight = edgeWeight)
    print("Write Files")
    if len(args.nxFileDir) > 0:
        nx.write_gexf(G, "OverallG.gexf")
    adjacencyMat = nx.convert_matrix.to_numpy_matrix(G)
    sparsity = np.sum(adjacencyMat != 0) / (adjacencyMat.shape[0] * adjacencyMat.shape[1])
    print(f"Sparsity: {sparsity}")
    np.savetxt(args.adjFile, adjacencyMat, delimiter=',')
    np.savetxt(args.featureFile, counts, delimiter=',', fmt='%d')
    np.savetxt('coordinate.csv', node, delimiter=',', fmt='%d')
    with open('time.csv', 'w') as f:
        for i in range(len(nodeTimeList)):
            print(getTimeStr(nodeTimeList[timeOrder[i]]), file=f)
    if len(args.adjFig) > 0:
        plt.imsave(args.adjFig, adjacencyMat)
    print("Done")