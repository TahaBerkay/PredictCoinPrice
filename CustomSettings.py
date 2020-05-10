import sys
from configparser import RawConfigParser, ConfigParser

file = 'settings.ini'

try:
    config = RawConfigParser()
    config.read(file)
    DEFAULT_DATASET = config.get('custom', 'DEFAULT_DATASET')
    DATE_FORMAT = config.get('custom', 'DATE_FORMAT')
    DATAFILES_DIR = config.get('custom', 'DATAFILES_DIR')
    WINDOW_SIZE = int(config.get('custom', 'WINDOW_SIZE'))
except ConfigParser.NoOptionError:
    print('could not read configuration file')
    sys.exit(1)
