import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, g):
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        g1 = self.up_sample(g1)  # Ensure g1 matches x1 in spatial dimensions
        psi = self.relu(x1 + g1)
        psi = self.psi(psi)
        return x * psi


class SEBlock(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale


class DenseConvBlock(nn.Module):

    def __init__(self, in_channels, growth_rate, layers):
        super(DenseConvBlock, self).__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for i in range(layers)
        ])

    def forward(self, x):
        features = [x]
        for block in self.blocks:
            out = block(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class ResidualConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)


class EnhancedUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(EnhancedUNet, self).__init__()

        self.enc1 = ResidualConvBlock(in_channels, 64)
        self.enc2 = ResidualConvBlock(64, 128)
        self.enc3 = ResidualConvBlock(128, 256)
        self.enc4 = ResidualConvBlock(256, 512)
        self.enc5 = ResidualConvBlock(512, 1024)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DenseConvBlock(1024, 128, layers=4)
        self.bottleneck_out_conv = nn.Conv2d(1024 + 4 * 128, 1024, kernel_size=1)

        self.dec5 = self._up_conv(1024, 512)
        self.dec4 = self._up_conv(512, 256)
        self.dec3 = self._up_conv(256, 128)
        self.dec2 = self._up_conv(128, 64)
        self.dec1 = self._up_conv(64, 32)

        self.skip_conv5 = nn.Conv2d(1024, 512, kernel_size=1)
        self.skip_conv4 = nn.Conv2d(512, 256, kernel_size=1)
        self.skip_conv3 = nn.Conv2d(256, 128, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(128, 64, kernel_size=1)
        self.skip_conv1 = nn.Conv2d(64, 32, kernel_size=1)

        self.se5 = SEBlock(512)
        self.se4 = SEBlock(256)
        self.se3 = SEBlock(128)
        self.se2 = SEBlock(64)
        self.se1 = SEBlock(32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    @staticmethod
    def _up_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        b = self.bottleneck(self.pool(e5))
        b = self.bottleneck_out_conv(b)

        d5 = self.dec5(b) + self.skip_conv5(e5)
        d5 = self.se5(d5)
        d4 = self.dec4(d5) + self.skip_conv4(e4)
        d4 = self.se4(d4)
        d3 = self.dec3(d4) + self.skip_conv3(e3)
        d3 = self.se3(d3)
        d2 = self.dec2(d3) + self.skip_conv2(e2)
        d2 = self.se2(d2)
        d1 = self.dec1(d2) + self.skip_conv1(e1)
        d1 = self.se1(d1)

        return self.final(d1)
