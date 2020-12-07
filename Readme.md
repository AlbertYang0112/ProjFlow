# ProjFlow

Traffic flow prediction. This work is based on the:

    Bing Yu*, Haoteng Yin*, Zhanxing Zhu. Spatio-temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting. In Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI), 2018`


## Workflow

- Data Preprocessing
- Network Setup
- Train & Apply STGCN
- Inspect the Output

## Data Pre-processing

- Convert the WGS84 coordinates to local cartesian coordinates.
- Calculate the GeoHash block coordinates.
  - Three block coordinates, not mixed together.
- Filter pings with low accuracy.
- Filter the blocks whose GeoHash coordinates are smaller than the boundary.

- Usage: `python converter.py datasetDir outputDir [optional args]`, optional args contain:
  - `--size`: The GeoHash block size. If not specified, block sizes are set to 1000;
  - `--thread n`: Multiprocessing;
  - `--mapLbX --mapLbY`: The lower boundary of the map. If not specified, the boundary coordinates are set according to the Austin geometry.

## Network Setup
- Node Filtering
  - Collect **all** GeoHash blocks in the dataset and set node for each block.
    - If the #pings in the block is less than the threshold `--pingTh` for all time slices in the dataset, the block is skipped.
    - If the #users in the block is less than the threshold `--userTh` for all time slices in the dataset, the block is skipped.
    - If the average #users in all time slices is lower than the threshold `--avgNodeUserTh`, the block is skipped.
- Data slicing
  - Slices are specified with window size `--windowSize` and window interval `--interval`
    - For example, we have two contiguous time slices $(t_{s1}, t_{e1})$ and $(t_{s2}, t_{e2})$
    - $\text{windowSize } = |t_{e1} - t_{s1}| = |t_{e2} - t_{s2}|$
    - $\text{interval } = |t_{s2} - t_{s1}|$
- Connect neighboring GeoHash blocks iff their distance belows the threshold `--edgeTh`.
  - The weight of the edge is set to exp(-dist^2/sigma^2). Longer -> lower, controlled with `--sigma`
  - No self-loop
- Export the adjacency matrix and feature matrix:
  - Definition of *feature*: Stacked #pings in all time slices;
  - Shape: (#node, #node) and (#node, #time slices), respectively;
- Export the helper files:
  - *coordinate.csv*: The GeoHash coordinate for each node.
  - *time.csv*: The start time for each time slice.
  - The data orders of *coordinate.csv* and *time.csv* are the same with the adjacency matrix and feature matrix.
- (Optional) Export the GEXF network file:
  - Export the network for each time slice;
  - Export the network for the whole dataset;
- (Optional) Export the image for adjacency matrix
- Usage: `python networkSetup.py dataPathList adjacencyFileName featureFileName [optional args]`
  - `--interval`: Time interval between two contiguous time slices. Should be able to divide 24 * 60 exactly.
  - `--windowSize`: The length of the time slices.
  - `--edgeTh`: The maximum L2 **block** distance to connect two nodes. Unit: block.
  - `--sigma`: The scaling factor of edge weight.
  - `--pingTh`: The minimum #ping for setting a node. If the #pings of the node in all time slices are less than `--pingTh`, the node is skipped.
  - `--userTh`: The minimum #user for setting a node. If the #users of the node in all time slices are less than `--userTh`, the node is skipped.
  - `--averageNodeUserTh`: The minimum average #user for setting a node.
  - `--nxFileDir`: Directory to export all the GEXF network file.
  - `--adjFig`: File name of the image of the adjacency matrix.
  - `--mapLbX --mapLbY`: The lower boundary of the map. If not specified, the boundary coordinates are set according to the Austin geometry.
  - `--size`: The GeoHash block size. If not specified, block sizes are set to 1000;


## STGCN Model

- Usage `python main.py AdjMat Feature Interval --n_his --batch_size --epoch --save --ks --kt --lr --opt --cls --seed`
  - `AdjMat`: Adjacency matrix generated in network setup
  - `Feature`: Feature matrix generated in network setup
  - `Interval`: Interval between time slices.
  - `--n_his`: #Slices to be consumed to predict the future population density
  - `--batch_size`: Batch size
  - `--epoch`: Training epoch
  - `--save`: Save model after every '`--save`' iterations
  - `--ks`: The order of Laplacian approximation of graph convolution.
  - `--kt`: Temporal convolutional kernel size
  - `--lr`: Learning rate
  - `--opt`: "RMSProp" or "ADAM"
  - `--cls`: #Classes to be classified
  - `--seed`: Random seed.
- Output: Output is collected in `./output/$(cls)/models`
  - `STGCN-*`: The trained tensorflow model
  - `Bin.csv`: The LUT mapping feature (#users/#pings) -> classification
  - `testInput.csv`: The test partition of test dataset.
  - `testLabel.csv`: The ground truth of test classification
  - `testPred.csv`: The prediction of test classification
  - `trainError.csv`: The error for each time slice & node in training set.
  - `valError.csv`: The error for each time slice & node in valadition set.
  - `tensorboard/`: The tensorboard logdir.
- Input format:
  - `AdjMat`: The adjacency matrix
    - CSV format, with delimiter ",".
    - Shape (n, n), where n is the #nodes.
  - `Feature`: The feature matrix.
    - CSV format, with delimiter ",".
    - Shape (n, s), where n is the #nodes and s is the #slices.
- Output format: 
  - `testInput.csv`, `testLabel.csv`, `testPred.csv`: The input/ground truth/prediction of test set.
    - CSV format, with delimiter ",".
    - Shape (n, testS) where n is the #nodes, and testS is the #slices in test set.
  -  `trainError.csv`
    - CSV format, with delimiter ",".
    - Shape (n, trainS) where n is the #nodes, and trainS is the #slices in training set.
    - trainError[n, trainS] = the training error at node n and time slice trainS.
  - `valError.csv`: The same with `trainError.csv`, the dataset change from training set to validation set.

## Examine the Output

We provide an interactive example to visualize the output in `demo.ipynb`.

## Environment & Dependencies

We provide the `Dockerfile` and `devcontainer.json` in `.devcontainer/` to help you construct the environment automatically with docker and vscode.
To support this feature, you need to install docker with nvidia support, and vscode remote plugin. But you can also construct the environment by yourself.

Dependencies:
- Python 3.6
- numpy & scipy & pandas
- pyproj
- tqdm
- h5py
- networkx
- tensorflow & tensorboard
- sklearn
- jupyter & notebook
- opencv-python
