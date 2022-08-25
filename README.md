# OpenDRIVE to NetworkX and GeoPandas Converter

This project allows to convert an OpenDRIVE map to a `geopandas.GeoDataFrame` and intersections of the network in `NetworkX`. Note that only the OpenDRIVE reference lines are currently supported.

## DLR AIM Research Intersection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4043193.svg)](https://doi.org/10.5281/zenodo.4043193)

This project uses the OpenDRIVE representation of the DLR AIM research intersection which is available via [Zenodo](https://doi.org/10.5281/zenodo.4043193) as Creative Commons Attribution 3.0 Internation license. To run the examples, you need to download the OpenDRIVE file from zenodo. If you have installed this project, just run the following command

```bash
xodr2nxgpd-dlr-fetcher
```

## Dependencies

This work is based on several packages

* [`opendrive2lanelet`](https://gitlab.lrz.de/tum-cps/opendrive2lanelet). The OpenDRIVE file parsing module is utilized to convert road geometries.
* [`NetworkX`](https://networkx.org/)

* [`Pandas`](https://pandas.pydata.org/)
* [`GeoPandas`](https://geopandas.org/en/stable/)
