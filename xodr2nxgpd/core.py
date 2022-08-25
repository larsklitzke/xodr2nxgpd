import functools
import logging
from typing import Any

import networkx as nx
from lxml.etree import ElementTree, parse

from xodr2nxgpd.io import DLR_MAP_PATH


class TopologyMixin():

    def __init__(self, graph: nx.DiGraph, tree: ElementTree = None, *args, **kwargs) -> None:
        self._graph = graph
        self._tree = tree

        super().__init__(*args, **kwargs)

    @property
    def tree(self) -> ElementTree:
        return self._tree

    def next(self, node: int):
        return self._graph.successors(node)

    def before(self, node: int):
        return self._graph.predecessors(node)

    def __getitem__(self, key):
        return self._graph[key]

    def __getattr__(self, __name: str) -> Any:
        if __name == '_graph' and __name not in self.__dict__:
            # handle unpickling object
            raise AttributeError()

        return getattr(self._graph, __name)


class InvalidTree(BaseException):
    pass


class OpenDriveTree():

    @staticmethod
    def validate(tree: ElementTree):
        if tree.getroot().tag.lower() != 'opendrive':
            raise InvalidTree('The tree does not have an OpenDRIVE root node')
        return tree

    @staticmethod
    @functools.lru_cache()
    def from_file(file: str = DLR_MAP_PATH) -> ElementTree:

        logging.info('Load OpenDRIVE file from %s', file)

        return OpenDriveTree.validate(parse(file))
