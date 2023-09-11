import functools
import logging
import os
import tempfile
import zipfile

import requests
from lxml import etree
from opendrive2lanelet.opendriveparser.parser import parse_opendrive

from xodr2nxgpd import ROOT_PATH

DLR_OPENDRIVE_URL = 'https://zenodo.org/record/4043193/files/bs-inner-ring-road-v1.0.0.zip'

DLR_MAP_PATH = os.path.join(os.path.join(ROOT_PATH, 'bs-inner-ring-road-v1.0.0', 'bs-inner-ring-road.xodr'))

FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')


@functools.lru_cache
def parse_opendrive(tree: etree.ElementTree):

    return parse_opendrive(tree.getroot())


def load_intersection_selection_transformation() -> etree.XSLT:
    return etree.XSLT(etree.parse(os.path.join(ROOT_PATH, 'odr_intersection_selection.xml')))


def load_intersection_extraction_transformation() -> etree.XSLT:
    return etree.XSLT(etree.parse(os.path.join(ROOT_PATH, 'odr_intersection_extraction.xml')))


def fetch_dlr_brunswick_opendrive_map():

    logging.info('Fetching DLR AIM OpenDRIVE map from %s', "/".join(DLR_OPENDRIVE_URL.split('/')[:-1]))

    with tempfile.NamedTemporaryFile('w+b') as f:
        f.write(requests.get(DLR_OPENDRIVE_URL).content)

        with zipfile.ZipFile(f) as tempzip:
            tempzip.extractall(ROOT_PATH)
            logging.info('OpenDRIVE map available at %s', ROOT_PATH)
