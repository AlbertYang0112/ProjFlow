# ProjFlow

---

Traffic flow prediction.

## Data Pre-processing

- Convert the WGS84 coordinates to local cartesian coordinates.
- Calculate the GeoHash block coordinates.
  - Three block coordinates, not mixed together.
- Filter pings with low accuracy.
- ~~Fill the missing/invalid altitude with the average altitude.~~
- Filter the blocks whose GeoHash coordinates are smaller than the boundary.

- Usage: `python converter.py datasetDir outputDir [optional args]`, optional args contain:
  - `--size`: The GeoHash block size. If not specified, block sizes are set to 2000;
  - `--thread n`: Multiprocessing;
  - `--mapLbX --mapLbY`: The lower boundary of the map. If not specified, the boundary coordinates are set according to the minimum coordinate of each csv file.

## Network Setup

- Collect **all** GeoHash blocks in the dataset and set node for each block.
  - If the #pings in the block is less than a threshold `--pingTh` for all time slices in the dataset, the block is skipped.
- Split the dataset into time slices by `--interval` minutes.
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
- Usage: `python networkSetup.py preprocessedDatasetDir adjacencyFileName featureFileName [optional args]`
  - `--interval`: The duration of each time slice. Should be able to divide 24 * 60 exactly.
  - `--edgeTh`: The maximum L2 **block** distance to connect two nodes. Unit: block.
  - `--sigma`: The scaling factor of edge weight.
  - `--pingTh`: The minimum #ping for setting a node. If the #pings of the node in all time slices are less than `--pingTh`, the node is skipped.
  - `--nxFileDir`: Directory to export all the GEXF network file.
  - `--adjFig`: File name of the image of the adjacency matrix.


## Environment & Dependencies
- Python 3.6
- numpy
- pyproj
- pandas
- tqdm
