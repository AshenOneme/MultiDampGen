import torch
import torch.nn as nn
from torch.nn import functional as F

class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x += identity
        return F.gelu(x)

class SpecialBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SpecialBlock, self).__init__()
        self.change_channel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        return F.gelu(x)

class Discriminator(nn.Module):
    def __init__(self, classes_num):
        super(Discriminator, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv2d(4, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(3, 1, 1)
        )
        self.layer1 = nn.Sequential(
            CommonBlock(64, 64),
            CommonBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            SpecialBlock(64, 128),
            CommonBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256),
            CommonBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512),
            CommonBlock(512, 512)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, classes_num)
        )

    def forward(self, x):
        x = self.prepare(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = Discriminator(classes_num=64)
    sample_input = torch.randn(128, 4, 32, 32)
    output = model(sample_input)
    print(f'Output shape: {output.shape}')
