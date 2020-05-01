import sys
from configparser import RawConfigParser, ConfigParser

file = 'settings.ini'

try:
    config = RawConfigParser()
    config.read(file)
    DEFAULT_DATASET = config.get('custom', 'DEFAULT_DATASET')
    DATE_FORMAT = config.get('custom', 'DATE_FORMAT')
except ConfigParser.NoOptionError:
    print('could not read configuration file')
    sys.exit(1)
