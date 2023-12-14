import os
from pyannote.database import get_protocol, FileFinder, registry

class CowAudioDataLoader:
    def __init__(self, database_path):
        self.database_path = database_path
        self.protocol = None

    def load_dataset(self):
        registry.load_database(self.database_path)
        os.environ["PYANNOTE_DATABASE_CONFIG"] = self.database_path

    def set_protocol(self, protocol_name):
        self.protocol = get_protocol(protocol_name, preprocessors={"audio": FileFinder()})