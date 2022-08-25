import sys
import os
import logging

logging.basicConfig()

if sys.prefix in os.path.abspath(__file__):
    # the package was installed
    logging.info('Prepare to run in deployment mode')
    ROOT_PATH = os.path.join(sys.prefix, 'etc', os.path.basename(os.path.dirname(__file__)))

    if not os.path.exists(ROOT_PATH):
        ROOT_PATH = os.path.join(sys.prefix, 'local', 'etc', os.path.basename(os.path.dirname(__file__)))

    if not os.path.exists(ROOT_PATH):
        raise RuntimeError('Failed to find tasi supplementary data in system prefix %s.', sys.prefix)

else:
    logging.info('Prepare to run in development mode')
    # we are in development mode
    ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'etc')
