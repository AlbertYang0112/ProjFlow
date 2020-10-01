# ProjFlow

---

Traffic flow prediction.

## Data Pre-processing

- Convert the WGS84 coordinates to local cartesian coordinates.
- Calculate the GeoHash block coordinates.
  - Three block coordinates, not mixed together.
- Filter pings with low accuracy.
- ~~Fill the missing/invalid altitude with the average altitude.~~

- Usage: `python converter.py datasetDir outputDir [optional args]`, optional args contain:
  - `--size[X-Z]`: The GeoHash block size. If not specified, block sizes are set to 2000;
  - `--thread n`: Multiprocessing;
  - `--mapLb[X-Z]`: The lower boundary of the map. If not specified, the boundary coordinates are set according to the minimum coordinate of each csv file.

## Environement & Dependencies
- Python 3.6
- numpy
- pyproj
- pandas
- tqdm
