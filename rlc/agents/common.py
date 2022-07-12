from typing import List
from typing import Type

from torch import nn
from torch import Tensor


class CNN(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        kernel_size: int = 3,
        activation: Type[nn.Module] = nn.ReLU,
        last_activation: Type[nn.Module] = nn.Identity,
        batch_norm: bool = False,
    ):
        super().__init__()
        channels = [input_channels] + hidden_channels + [out_channels]
        layers = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            if batch_norm:
                layers += [nn.BatchNorm2d(in_c)]
            layers += [nn.Conv2d(in_c, out_c, kernel_size), activation()]
        layers[-1] = last_activation()
        super().__init__(*layers)


class MLP(nn.Sequential):
    def __init__(
        self,
        input_features: int,
        hidden_features: List[int],
        out_features: int,
        activation: Type[nn.Module] = nn.ReLU,
        last_activation: Type[nn.Module] = nn.Identity,
        batch_norm: bool = False,
    ):
        super().__init__()
        features = [input_features] + hidden_features + [out_features]
        layers = []
        for in_f, out_f in zip(features[:-1], features[1:]):
            if batch_norm:
                layers += [nn.BatchNorm1d(in_f)]
            layers += [nn.Linear(in_f, out_f), activation()]
        layers[-1] = last_activation()
        super().__init__(*layers)


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        conv_hiddens: List[int],
        conv_to_mpl: int,
        kernel_size: int = 3,
        activation: Type[nn.Module] = nn.ReLU,
        last_activation: Type[nn.Module] = nn.Identity,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.cnn = CNN(
            input_channels,
            conv_hiddens,
            conv_to_mpl,
            kernel_size,
            activation,
            last_activation,
            batch_norm,
        )
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = self.avg(x)
        x = x.reshape(x.size(0), -1)
        return x
