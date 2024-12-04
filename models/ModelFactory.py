from modules.Unetbase import Unetbase
from modules.FourierUnet import FourierUnet
from modules.SimpleCNN import SimpleCNN

class ModelFactory:
    models = {
        'FourierUnet': FourierUnet,
        'Unetbase' : Unetbase,
        'SimpleCNN' : SimpleCNN
    }

    @staticmethod
    def create_model(name, **kwargs):
        if name in ModelFactory.models:
            return ModelFactory.models[name](**kwargs)
        else:
            raise ValueError(f"Model {name} not found in the registry.")
