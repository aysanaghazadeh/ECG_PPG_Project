from .unet import UNet
from torch import nn
from .config import Config


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.config = config
        self.models = {
            'U_Net': UNet
        }
        self.model = self.models[self.config.model](self.config)

    def forward(self, x):
        return self.model(x)

















