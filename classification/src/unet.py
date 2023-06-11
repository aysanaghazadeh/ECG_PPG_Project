import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, MaxPool1d, ReLU, Linear
from torch.nn import functional as F
from torchvision.transforms import CenterCrop


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.ReLU = ReLU()

    def forward(self, x):
        conv1_output = self.conv1(x)
        conv2_input = self.ReLU(conv1_output)
        conv2_output = self.conv2(conv2_input)
        output = self.ReLU(conv2_output)
        return output


class LastBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LastBlock, self).__init__()
        self.conv1 = Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.conv2 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.conv3 = Conv1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, padding=1)
        self.ReLU = ReLU()

    def forward(self, x):
        conv1_output = self.conv1(x)
        conv2_input = self.ReLU(conv1_output)
        conv2_output = self.conv2(conv2_input)
        conv3_input = self.ReLU(conv2_output)
        output = self.conv3(conv3_input)
        return output


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        channels = [1, 64, 128, 256, 512, 1024]
        self.blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(0, len(channels) - 1)]
        )
        self.pooling_layer = MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
            x = self.pooling_layer(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        channels = [1024, 512, 256, 128, 64, 1]
        self.up_convs = nn.ModuleList(
            [ConvTranspose1d(channels[i], channels[i + 1], kernel_size=2, stride=2)
             for i in range(0, len(channels) - 2)]
        )
        self.blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(0, len(channels) - 3)]
        )
        self.last_block = LastBlock(channels[len(channels) - 3],  # in_channel
                                      channels[len(channels) - 2],  # hidden_channel
                                      channels[len(channels) - 1])  # out_channel

    def forward(self, encoder_output):
        x = encoder_output[len(encoder_output) - 1]  # get the output of the last layer of the encoder
        x = self.up_convs[0](x)
        for i, block in enumerate(self.blocks):
            enc_features = encoder_output[len(encoder_output) - i - 2]  # output of the same layer in encoder
            enc_features = self.copy_and_crop(x, enc_features)
            x = torch.cat([x, enc_features], dim=1)
            x = block(x)
            x = self.up_convs[i + 1](x)
        enc_features = encoder_output[0]
        enc_features = self.copy_and_crop(x, enc_features)
        x = torch.cat([x, enc_features], dim=1)

        return self.last_block(x)

    @staticmethod
    def copy_and_crop(x, enc_features):
        (_, H, W) = x.shape
        enc_features = CenterCrop([H, W])(enc_features)
        return enc_features


class UNet(nn.Module):
    def __init__(self, config, retain_dim=True):
        super(UNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = Linear(in_features=994, out_features=1)
        self.retain_dim = retain_dim
        self.config = config

    def forward(self, x):
        output_size = x.shape
        encoder_outputs = self.encoder(x)
        output = self.decoder(encoder_outputs)
        # if self.retain_dim:
        #     output = F.interpolate(output, output_size)
        output = self.classifier(output)
        output = torch.squeeze(output)
        return output

