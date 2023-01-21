import functools
import logging
from enum import IntEnum
from typing import Any, List, Tuple, Union
from xml.etree import ElementTree

import geopandas
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import LineString

from xodr2nxgpd.core import OpenDriveTree, TopologyMixin
from xodr2nxgpd.io import load_intersection_selection_transformation
from xodr2nxgpd.opendriveparser.elements.opendrive import OpenDrive
from xodr2nxgpd.opendriveparser.elements.road import Road as OdrRoad


class SpecialNode(IntEnum):
    START = -1
    END = -2
    MISSING = -3

    @classmethod
    def isa(cls, other):
        try:
            return other in cls
        except TypeError:
            return other in [o.value for o in cls]


class IntersectionRoadType(IntEnum):
    Incoming = 0
    Connecting = 1
    Outgoing = 2


def get_road_attributes(road: ElementTree.Element) -> Tuple[int, int, int]:

    # verify id is valid
    try:
        id_ = int(road.get('id'))
    except ValueError:
        id_ = SpecialNode.MISSING

    # verify successor is valid
    try:
        successor_ = int(road.get('successor'))
    except ValueError:
        successor_ = SpecialNode.MISSING

    try:
        predecessor_ = int(road.get('predecessor'))
    except ValueError:
        predecessor_ = SpecialNode.MISSING

    return id_, predecessor_, successor_


class IntersectionNetwork(TopologyMixin):

    def __init__(self, intersection_id: int, *args, **kwargs) -> None:
        self._intersection_id = intersection_id

        super().__init__(*args, **kwargs)

    @property
    def intersection_id(self) -> int:
        return self._intersection_id

    @classmethod
    def init_network(cls, tree: ElementTree):

        G = nx.DiGraph()

        edges = []

        for road in tree.findall('connectivity/node'):

            # fail-safe access on road attributes
            id_, predecessor_, successor_ = get_road_attributes(road)

            # if no predecessor available
            if predecessor_ == SpecialNode.MISSING:
                # link this node to the start node
                edges.extend([(SpecialNode.START, id_)])
            else:
                # otherwise, link the predecessor to the start node and to this node
                edges.extend([(SpecialNode.START, predecessor_), (predecessor_, id_)])

            # if no successor available
            if successor_ == SpecialNode.MISSING:
                # link this node to the end node
                edges.append((id_, SpecialNode.END))
            else:
                # link this node so the successor and the successor to the end node
                edges.extend([(id_, successor_), (successor_, SpecialNode.END)])

        G.add_edges_from(edges)

        return G

    @property
    def roads(self) -> List[int]:
        return list(set(self.nodes) - set(SpecialNode))

    @property
    def incoming_roads(self) -> List[int]:
        return list(self.next(SpecialNode.START))

    @property
    def outgoing_roads(self) -> List[int]:
        return list(self.before(SpecialNode.END))

    @property
    def connecting_roads(self) -> List[int]:
        roads = []

        for r in self.next(SpecialNode.START):
            roads.extend(list(self.next(r)))
        return roads

    def road_type(self, road: int) -> IntersectionRoadType:

        if road in self.incoming_roads:
            return IntersectionRoadType.Incoming

        if road in self.outgoing_roads:
            return IntersectionRoadType.Outgoing

        if road in self.connecting_roads:
            return IntersectionRoadType.Connecting

        raise ValueError(f'The road {road} has an unsupported type')

    @classmethod
    def get_topology(cls, tree: ElementTree, intersection_id: Union[str, int]) -> ElementTree:

        logging.info('Applying XLST transformation to get network topology')
        transform = load_intersection_selection_transformation()

        return transform(tree, junction=str(intersection_id))

    @classmethod
    def from_tree(cls, intersection_id: Union[int, str], tree: OpenDriveTree, *args, **kwargs):

        topology = cls.get_topology(tree, intersection_id=intersection_id)

        # instantiate the intersection
        obj = cls(intersection_id=intersection_id, graph=cls.init_network(topology), tree=tree)

        return obj

    @classmethod
    def from_file(cls, intersection_id: int, file: str = None):

        # load the OpenDRIVE file
        tree = OpenDriveTree.from_file(file)

        return cls.from_tree(tree=tree, intersection_id=intersection_id)

    @functools.lru_cache()
    def _parse_opendrive(self) -> OpenDrive:

        from xodr2nxgpd.io import parse_opendrive

        return parse_opendrive(self._tree)

    def as_geopandas(self, interpolation: float = 1) -> geopandas.GeoDataFrame:

        # get the OpenDRIVE representation of the loaded intersection
        odr = self._parse_opendrive()

        # parse the roads of the intersection and convert as GeoDataFrame
        gpf = pd.concat([
            IntersectionRoad.from_opendrive(odr.getRoad(road), self, interpolation=interpolation) for road in self.roads
        ],
                        axis=0).sort_index()

        # set the CRS as specified in the header
        if odr.header.georeference:
            gpf.crs = pyproj.CRS(odr.header.georeference)
        else:
            logging.info('No CRS defined in the OpenDRIVE map. Will use local coordinate system.')

        return gpf

    def plot(self, ax=None, dx=0.3, highlight_nodes=None, highlight_edges=None):

        if highlight_nodes is None:
            highlight_nodes = []

        if highlight_edges is None:
            highlight_edges = []

        pos = nx.shell_layout(self._graph)

        pos[SpecialNode.START] = np.array([-2, 0])
        pos[SpecialNode.END] = np.array([2, 0])

        # set positions of roads in the graph
        for road, x_offset in zip((self.incoming_roads, self.connecting_roads, self.outgoing_roads), (-1, 0, 1)):
            y = np.linspace(-1, 1, len(road))

            for r, y_ in zip(road, y):
                pos[r] = np.array([x_offset, y_])

        nx.draw_networkx_nodes(
            self._graph, pos, node_size=0, ax=ax, nodelist=list(set(list(self._graph.nodes)) - set(list(SpecialNode)))
        )

        # draw special nodes
        nx.draw_networkx_nodes(
            self._graph, pos, node_size=100, ax=ax, nodelist=[SpecialNode.START, SpecialNode.END], node_color="#3787c3"
        )

        # draw the highlighted nodes
        nx.draw_networkx_labels(
            self._graph,
            pos,
            clip_on=False,
            font_color="#AA0000",
            ax=ax,
            labels={p: p
                    for p in pos.keys()
                    if p in highlight_nodes and not SpecialNode.isa(p)}
        )

        # draw the normal nodes
        nx.draw_networkx_labels(
            self._graph,
            pos,
            clip_on=False,
            font_color="#000000",
            ax=ax,
            labels={p: p
                    for p in pos.keys()
                    if p not in highlight_nodes and not SpecialNode.isa(p)}
        )

        # draw start -> incoming roads
        edges_pos = pos.copy()

        for r in self.incoming_roads:
            edges_pos[r] = pos[r] - np.array([dx, 0])

        # draw highlighted edges
        nx.draw_networkx_edges(
            self._graph,
            edges_pos,
            ax=ax,
            node_size=0,
            min_target_margin=0,
            min_source_margin=5,
            edge_color="#AA0000",
            edgelist=list(set(self._graph.edges(SpecialNode.START)).intersection(highlight_edges))
        )

        # draw non-highlighted edges
        nx.draw_networkx_edges(
            self._graph,
            edges_pos,
            ax=ax,
            node_size=0,
            min_target_margin=0,
            min_source_margin=5,
            edgelist=list(set(self._graph.edges(SpecialNode.START)) - set(highlight_edges))
        )

        # draw incoming -> connecting
        edges_pos = pos.copy()

        for r in self.incoming_roads:
            edges_pos[r] = pos[r] + np.array([dx, 0])

        for r in self.connecting_roads:
            edges_pos[r] -= np.array([dx, 0])

        # draw highlighted edges
        nx.draw_networkx_edges(
            self._graph,
            edges_pos,
            ax=ax,
            node_size=0,
            min_target_margin=0,
            min_source_margin=5,
            edge_color="#AA0000",
            edgelist=list(set(self._graph.edges(self.incoming_roads)).intersection(highlight_edges))
        )

        # draw non-highlighted edges
        nx.draw_networkx_edges(
            self._graph,
            edges_pos,
            ax=ax,
            node_size=0,
            min_target_margin=0,
            min_source_margin=5,
            edgelist=list(set(self._graph.edges(self.incoming_roads)) - set(highlight_edges))
        )

        # draw connecting -> outgoing roads
        edges_pos = pos.copy()
        for r in self.connecting_roads:
            edges_pos[r] += np.array([2 * dx, 0])

        for r in self.outgoing_roads:
            edges_pos[r] -= np.array([dx * 0.5, 0])

        
        # draw highlighted edges
        nx.draw_networkx_edges(
            self._graph,
            edges_pos,
            ax=ax,
            node_size=0,
            min_target_margin=10,
            min_source_margin=5,
            edge_color="#AA0000",
            edgelist=list(set(self._graph.edges(self.connecting_roads)).intersection(highlight_edges))
        )

        # draw non-highlighted edges
        nx.draw_networkx_edges(
            self._graph,
            edges_pos,
            ax=ax,
            node_size=0,
            min_target_margin=10,
            min_source_margin=5,
            edgelist=list(set(self._graph.edges(self.connecting_roads)) - set(highlight_edges))
        )

        # draw outgoing -> end node
        edges_pos = pos.copy()
        for r in self.outgoing_roads:
            edges_pos[r] = pos[r] + np.array([1.5 * dx, 0])

    
        # draw highlighted edges
        nx.draw_networkx_edges(
            self._graph,
            edges_pos,
            ax=ax,
            node_size=0,
            min_target_margin=10,
            min_source_margin=5,
            edge_color="#AA0000",
            edgelist=list(set(self._graph.edges(self.outgoing_roads)).intersection(highlight_edges))
        )

        # draw non-highlighted edges
        nx.draw_networkx_edges(
            self._graph,
            edges_pos,
            ax=ax,
            node_size=0,
            min_target_margin=10,
            min_source_margin=5,
            edgelist=list(set(self._graph.edges(self.outgoing_roads)) - set(highlight_edges))
        )

class IntersectionRoad(geopandas.GeoDataFrame):

    _metadata = ['_intersection', *geopandas.GeoDataFrame._metadata]

    @property
    def _constructor(self):
        return IntersectionRoad

    @classmethod
    def from_opendrive(
        cls, road: OdrRoad, intersection: IntersectionNetwork, interpolation: float = 1
    ) -> "IntersectionRoad":
        """Create from an odr road

        Args:
            road (OdrRoad): The OpenDRIVE road
            intersection (IntersectionNetwork): The intersection this road belongs to
            interpolation (float, optional): The interpolation step size. Defaults to 1 meter.

        Returns:
            IntersectionRoad: The OpenDRIVE road as GeoPandas GeoDataFrame
        """
        # generate sampling points based on the length of the road and the interpolation precision
        x = np.linspace(0, road.planView.length, int(road.planView.length / interpolation))

        # sample from the road
        df = pd.DataFrame(list(map(lambda l: np.hstack(road.planView.calc(l)), x)), columns=['x', 'y', 'heading'])

        obj = cls([[
            LineString(geopandas.points_from_xy(df.x, df.y)).simplify(0.1),
            intersection.road_type(road.id),
            list(intersection.predecessors(road.id)),
            list(intersection.successors(road.id))
        ]],
                  index=pd.Index([road.id], name='id'),
                  columns=['geometry', 'road_type', 'predecessor', 'successor'])
        obj.intersection = intersection.intersection_id

        return obj
