import torch
import torch.nn as nn

class Iteration(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        ) 

    def forward(self, x):
        return self.block(x)

# class Encoder(Module):

# class Decoder():

class U_net(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, features=[64, 128, 256, 512]):
        super(U_net, self).__init__()
        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encode
        for feature in features:
            self.encoders.append(Iteration(input_channels, feature))
            input_channels = feature

        # Decode
        for feature in reversed(features):
            self.decoders.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.decoders.append(Iteration(feature*2, feature))

        # Bottleneck
        self.bottleneck = Iteration(features[-1], features[-1]*2)

        # Last Conv
        self.last_conv = nn.Conv2d(features[0], output_channels, kernel_size=1)

    def forward(self, x):
            
        # Encoder path (maxpooling):
        skip_connections = []
        for encode in self.encoders:
            x = encode(x)
            # Store all feature maps from encoder path for skip connection
            skip_connections.append(x)
            x = self.pool(x)
        skip_connections = skip_connections[::-1]

        # Bottleneck of U-net 
        x = self.bottleneck(x)

        # Decoder path (upconvolution & concat. with skip_connections):
        for index in range(0, len(self.decoders), 2):
            x = self.decoders[index](x)
            skip_connection = skip_connections[index//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoders[index+1](concat_skip)
            
        return self.last_conv(x)

# def test():
#     x = torch.randn((3, 1, 256, 256))
#     model = U_net(input_channels=1, output_channels=1)
#     preds = model(x)
#     print(preds.shape)
#     print(x.shape)
#     assert preds.shape == x.shape

# if __name__ == "__main__":
#     test()