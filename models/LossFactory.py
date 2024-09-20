from torch import nn
from modules.loss import ScaledLpLoss
from modules.loss import CustomMSELoss
from modules.loss import PearsonCorrelationScore

class LossFactory:
    @staticmethod
    def get_loss_function(name, **kwargs):  # 支持传递参数
        if name == "L1Loss":
            return nn.L1Loss()
        elif name == "L2Loss":
            return nn.MSELoss()
        elif name == "ScaledLpLoss":
            return ScaledLpLoss(**kwargs)  # p 和 reduction 参数可以通过 kwargs 传递
        elif name == "CustomMSELoss":
            return CustomMSELoss(**kwargs)  # reduction 参数可以通过 kwargs 传递
        elif name == "PearsonCorrelationScore":
            return PearsonCorrelationScore(**kwargs)  # channel 和 reduce_batch 参数可以通过 kwargs 传递
        else:
            raise ValueError(f"Unsupported loss function: {name}")
