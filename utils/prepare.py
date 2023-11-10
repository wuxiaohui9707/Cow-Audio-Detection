from pyannote.database import registry
from pyannote.database import get_protocol, FileFinder
import os

def database(path):
    registry.load_database(path)
    os.environ["PYANNOTE_DATABASE_CONFIG"] = path

def get_files(files,protocol):
    files = get_protocol(protocol, preprocessors={"audio": FileFinder()})
    return files