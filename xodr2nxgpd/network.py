import logging
from typing import List, Union

import geopandas
import networkx as nx
import pandas as pd
from lxml.etree import ElementTree

from xodr2nxgpd.core import OpenDriveTree, TopologyMixin
from xodr2nxgpd.intersection import IntersectionNetwork
from xodr2nxgpd.io import (DLR_MAP_PATH, load_intersection_extraction_transformation)


class RoadNetwork(TopologyMixin):

    def __init__(self, intersections: List[IntersectionNetwork], *args, **kwargs) -> None:
        self._intersections = {i.intersection_id: i
                               for i in intersections}

        super().__init__(*args, **kwargs)

    def intersection(self, intersection: Union[str, int]) -> IntersectionNetwork:
        return self._intersections[str(intersection)]

    @classmethod
    def get_topology(cls, tree: OpenDriveTree) -> ElementTree:

        logging.info('Applying XLST transformation to get network topology')
        transform = load_intersection_extraction_transformation()

        return transform(tree)

    @classmethod
    def from_tree(cls, tree: OpenDriveTree, topology: ElementTree = None, njobs=4):

        logging.info('Extract intersections from network')

        if topology is None:
            intersections = [
                IntersectionNetwork.from_tree(intersection_id=intersection.get('id'), tree=tree)
                for intersection in tree.findall('junction')
            ]
        else:
            intersections = [
                IntersectionNetwork(
                    intersection_id=intersection.get('id'),
                    graph=IntersectionNetwork.init_network(intersection),
                    tree=tree
                ) for intersection in topology.findall('intersection')
            ]

        graph = None

        for net in intersections:

            # merge graphs
            if graph is None:
                graph = net._graph
            else:
                graph = nx.compose(graph, net._graph)

        return cls(graph=graph, tree=tree, intersections=intersections)

    @classmethod
    def from_file(cls, file: str = DLR_MAP_PATH):

        # load the OpenDRIVE file
        tree = OpenDriveTree.from_file(file)

        return cls.from_tree(tree)

    def as_geopandas(self, interpolation: float = 0.25) -> geopandas.GeoDataFrame:

        return geopandas.GeoDataFrame(
            pd.concat([i.as_geopandas(interpolation=interpolation) for i in self._intersections.values()],
                      axis=0,
                      keys=self._intersections.keys(),
                      names=['intersection', 'road'])
        )
