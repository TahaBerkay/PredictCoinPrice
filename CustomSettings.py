import json
import sys
from configparser import RawConfigParser, ConfigParser

file = 'settings.ini'

try:
    config = RawConfigParser()
    config.read(file)
    DEFAULT_DATASET = config.get('custom', 'DEFAULT_DATASET')
    DATE_FORMAT = config.get('custom', 'DATE_FORMAT')
    DATAFILES_DIR = config.get('custom', 'DATAFILES_DIR')
    NB_JOBS_GRIDSEARCH = int(config.get('custom', 'NB_JOBS_GRIDSEARCH'))
    HOLD_RANGE = float(config.get('custom', 'HOLD_RANGE'))
    WINDOW_SIZE = int(config.get('custom', 'WINDOW_SIZE'))
    SHORT_PERIODS = int(config.get('custom', 'SHORT_PERIODS'))
    SIGNAL_PERIODS = int(config.get('custom', 'SIGNAL_PERIODS'))
    ALGORITHM_COMBINATIONS = json.loads(config.get('custom', 'ALGORITHM_COMBINATIONS'))['algorithms']
except ConfigParser.NoOptionError:
    print('could not read configuration file')
    sys.exit(1)
