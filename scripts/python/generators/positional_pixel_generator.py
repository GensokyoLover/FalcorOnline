import torch
import torch.nn as nn
from utils import *


def fc_layer(in_features, out_features):
    net = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(inplace=True)
    )

    return net


class PositionalPixelGenerator(nn.Module):

    def __init__(self, buffers_features, variables_features, out_features=3, hidden_features=700, hidden_layers=8, device='cuda', sine=False):
        super(PositionalPixelGenerator, self).__init__()

        self.buffers_features = buffers_features
        self.variables_features = variables_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features

        self.inner_pos = fc_layer(in_features=3, out_features=hidden_features)

        self.inner = fc_layer(in_features=hidden_features + buffers_features + variables_features, out_features=hidden_features)

        self.hidden = nn.ModuleList()
        for i in range(hidden_layers):
            self.hidden.append(fc_layer(in_features=(hidden_features + buffers_features + variables_features), out_features=hidden_features))

        self.outer = nn.Linear(in_features=hidden_features + buffers_features + variables_features, out_features=out_features)

        print("Number of model parameters:")
        print_network(self)

    def forward(self, input):
        print(input.shape)
        # Get emission and position from input
        emission = input[:, :, :, 0:3]
        position = input[:, :, :, 6:9]

        # Ignore emission, it's passed through to the output
        input = input[:, :, :, 3:]

        # Positional encoding
        x0 = self.inner_pos(position)
        x0 = torch.cat([x0, input], 3)

        x1 = self.inner(x0)
        prev = x1

        for i in range(len(self.hidden)):
            x2 = torch.cat([prev, input], 3)
            x2 = self.hidden[i](x2)
            prev = x2

        x2 = torch.cat([prev, input], 3)
        output = self.outer(x2)

        # Merge emission and predicted output
        output = torch.where(emission > 1.0, emission, output+emission)

        return output
    
# convert 4D input and output to 2D input output to moe
class MoePositionalPixelGenerator(nn.Module):

    def __init__(self, buffers_features, variables_features, out_features=3, hidden_features=700, hidden_layers=8, device='cuda', sine=False):
        super(MoePositionalPixelGenerator, self).__init__()

        self.buffers_features = buffers_features
        self.variables_features = variables_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features

        self.inner_pos = fc_layer(in_features=3, out_features=hidden_features)

        self.inner = fc_layer(in_features=hidden_features + buffers_features + variables_features, out_features=hidden_features)

        self.hidden = nn.ModuleList()
        for i in range(hidden_layers):
            self.hidden.append(fc_layer(in_features=(hidden_features + buffers_features + variables_features), out_features=hidden_features))

        self.outer = nn.Linear(in_features=hidden_features + buffers_features + variables_features, out_features=out_features)

        print("Number of model parameters:")
        print_network(self)

    def forward(self, input):
        # Get emission and position from input
        emission = input[:, 0:3]
        position = input[:, 6:9]

        # Ignore emission, it's passed through to the output
        input = input[:, 3:]
        # Positional encoding
        x0 = self.inner_pos(position)
        x0 = torch.cat([x0, input], 1)

        x1 = self.inner(x0)
        prev = x1

        for i in range(len(self.hidden)):
            x2 = torch.cat([prev, input], 1)
            x2 = self.hidden[i](x2)
            prev = x2

        x2 = torch.cat([prev, input], 1)
        output = self.outer(x2)

        # Merge emission and predicted output
        output = torch.where(emission > 1.0, emission, output+emission)

        return output
    
class MoePositionalPixelEncoder(nn.Module):

    def __init__(self, buffers_features, variables_features, out_features=3, hidden_features=700, hidden_layers=8, device='cuda', sine=False):
        super(MoePositionalPixelEncoder, self).__init__()

        self.buffers_features = buffers_features
        self.variables_features = variables_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        print(hidden_features)
        self.inner_pos = fc_layer(in_features=3, out_features=hidden_features)

        self.inner = fc_layer(in_features=hidden_features + buffers_features + variables_features, out_features=hidden_features)

        self.hidden = nn.ModuleList()
        for i in range(hidden_layers):
            self.hidden.append(fc_layer(in_features=(hidden_features + buffers_features + variables_features), out_features=hidden_features))

        print("Number of model parameters:")
        print_network(self)

    def forward(self, input):
        # Get emission and position from input
        emission = input[:, 0:3]
        position = input[:, 6:9]

        # Ignore emission, it's passed through to the output
        input = input[:, 3:]
        # Positional encoding
        x0 = self.inner_pos(position)
        x0 = torch.cat([x0, input], 1)

        x1 = self.inner(x0)
        prev = x1

        for i in range(len(self.hidden)):
            x2 = torch.cat([prev, input], 1)
            x2 = self.hidden[i](x2)
            prev = x2

        x2 = prev

        return x2
    
class MoePositionalPixelDecoder(nn.Module):

    def __init__(self, buffers_features, variables_features, out_features=3, hidden_features=700, hidden_layers=8, device='cuda', sine=False):
        super(MoePositionalPixelDecoder, self).__init__()

        self.buffers_features = buffers_features
        self.variables_features = variables_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.hidden = nn.ModuleList()
        for i in range(hidden_layers):
            self.hidden.append(fc_layer(in_features=(hidden_features + buffers_features + variables_features), out_features=hidden_features))
        self.outer = nn.Linear(in_features=hidden_features + buffers_features + variables_features, out_features=out_features)
        print("Number of model parameters:")
        print_network(self)

    def forward(self, input, latent_feature):
        # Get emission and position from input
        emission = input[:, 0:3]

        # Ignore emission, it's passed through to the output
        input = input[:, 3:]
        # Positional encoding
        prev = latent_feature

        for i in range(len(self.hidden)):
            x2 = torch.cat([prev, input], 1)
            print(x2.shape)
            x2 = self.hidden[i](x2)
            prev = x2

        x2 = torch.cat([prev, input], 1)
        output = self.outer(x2)

        # Merge emission and predicted output
        output = torch.where(emission > 1.0, emission, output+emission)

        return output
    
    
    
class MoePositionalPixelEncoderPosNo(nn.Module):

    def __init__(self, buffers_features, variables_features, out_features=3, hidden_features=700, hidden_layers=8, device='cuda', sine=False):
        super(MoePositionalPixelEncoderPosNo, self).__init__()

        self.buffers_features = buffers_features
        self.variables_features = variables_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.total_features = hidden_features + buffers_features + variables_features
        print(hidden_features)

        self.inner = fc_layer(in_features=hidden_features + buffers_features + variables_features, out_features=hidden_features)

        self.hidden = nn.ModuleList()
        for i in range(hidden_layers):
            self.hidden.append(fc_layer(in_features=(hidden_features + buffers_features + variables_features), out_features=hidden_features))

        print("Number of model parameters:")
        print_network(self)

    def forward(self,x0):
        # Get emission and position from input
        input = x0[:,self.total_features-self.buffers_features-self.variables_features:]
        print(input.shape)
        x1 = self.inner(x0)
        prev = x1

        for i in range(len(self.hidden)):
            x2 = torch.cat([prev, input], 1)
            x2 = self.hidden[i](x2)
            prev = x2

        x2 = prev

        return x2