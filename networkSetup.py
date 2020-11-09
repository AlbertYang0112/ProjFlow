from argparse import RawDescriptionHelpFormatter
import os
from platform import node
import sys
from pyproj import Proj, Transformer, CRS
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
ADJ_FIG_GAMMA = 0.25
AUSTIN_PROJ4_DESC = "+proj=lcc \
    +lat_1=31.8833333333333 +lat_2=30.1166666666667 +lat_0=29.6666666666667 \
    +lon_0=-100.333333333333 +x_0=2296583.333333 +y_0=9842500 \
    +datum=NAD83 +units=m +no_defs"
DAY_LENGTH_MIN = 24 * 60
crsWGS84 = CRS.from_epsg(4326)
crsAustin = CRS.from_proj4(AUSTIN_PROJ4_DESC)
transformer = Transformer.from_crs(crsWGS84, crsAustin)
iTransformer = Transformer.from_crs(crsAustin, crsWGS84)

parser = argparse.ArgumentParser(description="Generate The Map")
parser.add_argument('dataFile', type=str, metavar="DataFile", help="Preprocessed Dataset")
parser.add_argument('adjFile', type=str, metavar="AdjFile", help='Adjacency Matrix File')
parser.add_argument('featureFile', type=str, metavar="FeatureFile", help='Feature File')
parser.add_argument("--interval", type=int, metavar='interval', help='interval between slices (minute)', default=-1)
parser.add_argument("--windowSize", type=float, help="Length of the slice (minute)", default=-1)
parser.add_argument('--edgeTh', type=int, metavar="nodeTh", help="Edge Filtering Threshold", default=5)
parser.add_argument('--sigma', type=float, metavar="Sigma", help="Edge Weight Scaling Factor", default=1)
parser.add_argument('--pingTh', type=int, metavar="pingTh", help="Edge Filtering Threshold", default=10)
parser.add_argument('--userTh', type=int, metavar="userTh", help="User Filtering Threshold", default=2)
parser.add_argument('--avgNodeUserTh', type=int, metavar="userTh", help="User Filtering Threshold", default=2)
parser.add_argument('--nxFileDir', type=str, metavar="nxFileDir", help="Path to dump the networkx file.", default='')
parser.add_argument('--adjFig', type=str, metavar="adjFig", help="Path to save the figure of adjacency matrix.", default='')
parser.add_argument("--mapLbX", type=float, default=30.18)
parser.add_argument("--mapLbY", type=float, default=-97.92)
parser.add_argument("--size", type=float, default=1000)

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

def dataSplit(dataDay1, dataDay2, interval, windowSize):
    baseUnixStamp = (dataDay1.loc[0, 'location_at'] + DATA_TIME_OFFSET) // DAY_LENGTH * DAY_LENGTH - DATA_TIME_OFFSET
    if dataDay2 is not None:
        baseUnixStamp2 = (dataDay2.loc[0, 'location_at'] + DATA_TIME_OFFSET) // DAY_LENGTH * DAY_LENGTH - DATA_TIME_OFFSET
        assert baseUnixStamp2 - baseUnixStamp == DAY_LENGTH_MIN * 60
    splitCnt = int(DAY_LENGTH_MIN / interval)
    idxList = []
    # print(baseUnixStamp)
    for splitIdx in range(splitCnt):
        timeStampLb = splitIdx * interval * 60 + baseUnixStamp
        timeStampUb = timeStampLb + windowSize * 60
        # print(f"Split {splitIdx}: {timeStampLb} - {timeStampUb} {getAustinTime(timeStampLb)} {getAustinTime(timeStampUb)}")
        sel = np.logical_and(dataDay1['location_at'] >= timeStampLb, dataDay1['location_at'] < timeStampUb)
        idxDay1 = dataDay1[sel].index
        if dataDay2 is not None and splitIdx * interval + windowSize > DAY_LENGTH_MIN:
            selRoll = dataDay2['location_at'] < timeStampUb
            idxDay2 = dataDay2[selRoll].index
            # print(f"Roll Split {splitIdx}: {getAustinTime(rollTimeStampLb)}-{getAustinTime(rollTimeStamlUb)}")
            idxList.append((idxDay1, idxDay2, getAustinTime(timeStampLb)))
        else:
            idxList.append((idxDay1, None, getAustinTime(timeStampLb)))

    return idxList

def setWindowParam(args):
    if args.windowSize == -1 and args.interval == -1:
        print("Set both window size & interval to default 3hrs")
        windowSize = DEFAULT_TIME_INTERVAL
        interval = DEFAULT_TIME_INTERVAL
    elif args.windowSize == -1:
        windowSize = args.interval
        interval = args.interval
    elif args.interval == -1:
        windowSize = args.windowSize
        interval = args.windowSize
    else:
        windowSize = args.windowSize
        interval = args.interval
    if (DAY_LENGTH_MIN - windowSize) % interval != 0:
        print(f"Day cannot be splitted into slices with window size {windowSize} min, interval {interval} min.")
        interval = (DAY_LENGTH_MIN - windowSize) / int((DAY_LENGTH_MIN-windowSize) / interval)
        print(f"Using interval = {interval}mins instead")
    return windowSize, interval


if __name__ == '__main__':
    args = parser.parse_args()
    windowSize, interval = setWindowParam(args)
    # Find all the CSV files
    csvFiles = []
    with open(args.dataFile) as f:
        for l in f.readlines():
            l = l.strip()
            pathSeg = l.split("/")
            p = "/".join(pathSeg[:-1])
            f = pathSeg[-1]
            csvFiles.append((p, f))
    print(f"Found {len(csvFiles)} files under {args.dataFile}")
    # fileLoader = tqdm(csvFiles)
    nodeList = []
    nodeCountList = []
    userCountList = []
    nodeTimeList = []
    if len(args.nxFileDir) > 0:
        os.makedirs(args.nxFileDir, exist_ok=True)
    # Collect all the nodes and ping counts
    for fileIdx in tqdm(range(len(csvFiles))):
        p, f = csvFiles[fileIdx]
        filePath1 = os.path.join(p, f)
        rawData1 = pd.read_csv(filePath1)
        rawData2 = None
        if fileIdx + 1 < len(csvFiles):
            p2, f2 = csvFiles[fileIdx + 1]
            filePath2 = os.path.join(p2, f2)
            rawData2 = pd.read_csv(filePath2)
        dataIdx = dataSplit(rawData1, rawData2, interval, windowSize)
        for idxDay1, idxDay2, t in dataIdx:
            dataSlice = rawData1.loc[idxDay1]
            if idxDay2 is not None:
                dataSliceDay2 = rawData2.loc[idxDay2]
                dataSlice = pd.concat((dataSlice, dataSliceDay2), axis=0, ignore_index=True)
            nodes, nodeIdx, nodeCount = np.unique(
                dataSlice.loc[:, ("bx", "by")], axis=0, 
                return_counts=True, return_inverse=True
            )
            userCnt = np.zeros_like(nodeCount)
            for i in range(len(nodeCount)):
                userCnt[i] = len(np.unique(dataSlice.loc[nodeIdx == i, "advertiser_id"]))
            validNodeIdx = np.logical_and(nodeCount > args.pingTh, userCnt > args.userTh)
            validNodes = nodes[validNodeIdx]
            validNodeCounts = nodeCount[validNodeIdx]
            nodeList.append(validNodes)
            nodeCountList.append(validNodeCounts)
            userCountList.append(userCnt)
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
    featureUser = np.zeros((nodeCnt, featureCnt), dtype=np.int)
    featurePing = np.zeros((nodeCnt, featureCnt), dtype=np.int)
    timeOrder = np.argsort(nodeTimeList)
    print(f"Collect the features")
    for i in tqdm(range(featureCnt), total=featureCnt):
        featureIdx = timeOrder[i]
        for n, userCnt, pingCnt in zip(nodeList[featureIdx], userCountList[featureIdx], nodeCountList[featureIdx]):
            idx = np.where(np.all(node == n, axis=1))
            assert len(idx) == 1, f"Node: {n}, where: {idx}"
            idx = idx[0]
            featureUser[idx, i] += userCnt
            featurePing[idx, i] += pingCnt
    # Filter the sparse node
    avgFeatureUser = np.average(featureUser, axis=1)
    validNode = avgFeatureUser > args.avgNodeUserTh
    node = node[validNode, :]
    nodeHash = nodeHash[validNode]
    featureUser = featureUser[validNode, :]
    featurePing = featurePing[validNode, :]
    print(f"Filtered {np.sum(~validNode)} sparse nodes")
    # Get the map boundary
    mapLbCoord = transformer.transform(args.mapLbX, args.mapLbY)
    nodeCoord = np.zeros_like(node, dtype=np.float)
    # Construct the Graph
    print("Construct the Graph")
    G = nx.Graph()
    nodeWeight = np.sum(featureUser, axis=1)
    nodeHashSet = set(nodeHash)
    for idx, (x, y) in tqdm(enumerate(node)):
        blkCoord = np.array([x, y]) * args.size + args.size / 2 + mapLbCoord
        nodeCoord[idx, :] = iTransformer.transform(blkCoord[0], blkCoord[1])
        G.add_node(
            nodeHash[idx], 
            weight=nodeWeight[idx], 
            viz={'position': {'x': x, 'y': y, 'z': 0}}, 
            latitude=nodeCoord[idx, 0], longitude=nodeCoord[idx, 1]
        )
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
    adjacencyMat = nx.convert_matrix.to_numpy_matrix(G)
    sparsity = np.sum(adjacencyMat < 1e-8) / (adjacencyMat.shape[0] * adjacencyMat.shape[1])
    print(f"Sparsity: {sparsity}")
    np.savetxt(args.adjFile, adjacencyMat, delimiter=',')
    np.savetxt(args.featureFile, featureUser, delimiter=',', fmt='%d')
    exportFileNamePrefix = '.'.join(args.featureFile.split('.')[:-1])
    np.savetxt(exportFileNamePrefix + "Ping.csv", featurePing, delimiter=',', fmt='%d')
    np.savetxt(exportFileNamePrefix + "Cord.csv", np.concatenate((node, nodeCoord), axis=1), delimiter=',', fmt="%d, %d, %.18f, %.18f")
    with open(exportFileNamePrefix + 'Time.csv', 'w') as f:
        for i in range(len(nodeTimeList)):
            print(getTimeStr(nodeTimeList[timeOrder[i]]), file=f)
    # if len(args.nxFileDir) > 0:
    nx.write_gexf(G, exportFileNamePrefix + "OverallG.gexf")
    if len(args.adjFig) > 0:
        plt.set_cmap('RdBu_r')
        plt.figure(figsize=(8, 8))
        plt.title(f"EdgeTh: {args.edgeTh} Sparsity: {np.round(sparsity * 100, 2)}%")
        plt.axis('off')
        plt.imshow(np.power(adjacencyMat, ADJ_FIG_GAMMA))
        plt.savefig(args.adjFig)
    print("Done")
