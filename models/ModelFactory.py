from modules.Unetbase import Unetbase
from modules.Unet import Unet
from modules.Unet2015 import Unet2015
from modules.Resnet import ResNet
from modules.FourierUnet import FourierUnet

class ModelFactory:
    models = {
        'FourierUnet': FourierUnet,
        'ResNet' : ResNet,
        'Unet' : Unet,
        'Unetbase' : Unetbase,
        'Unet2015' : Unet2015
    }

    @staticmethod
    def create_model(name, **kwargs):
        if name in ModelFactory.models:
            return ModelFactory.models[name](**kwargs)
        else:
            raise ValueError(f"Model {name} not found in the registry.")
