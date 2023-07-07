import torch
from torch.nn import Transformer, Embedding, Linear
import torch.nn as nn
from .unet import UNet


class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config
        self.linear_in = nn.Linear(self.config.input_size, 256)
        self.transformer = Transformer(
            d_model=256,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=256,
            dropout=0.2
        )
        self.fc = Linear(256, self.config.output_size)

    def forward(self, x):
        x = self.linear_in(x)
        x = x.permute(1, 0, 2)  # Reshape to (sequence_length, batch_size, hidden_size)
        output = self.transformer(x, x)
        output = output.permute(1, 0, 2)  # Reshape back to (batch_size, sequence_length, hidden_size)
        output = self.fc(output)
        return output


class TransformerModelWithUNetBackbone(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config
        self.unet = UNet(config)
        self.embedding = Embedding(self.config.input_size, 256)
        self.transformer = Transformer(
            d_model=256,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=256,
            dropout=0.2
        )
        self.fc = Linear(256, self.config.output_size)

    def forward(self, x):
        x = self.unet(x)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Reshape to (sequence_length, batch_size, hidden_size)
        output = self.transformer(x, x)
        output = output.permute(1, 0, 2)  # Reshape back to (batch_size, sequence_length, hidden_size)
        output = self.fc(output)
        return output
