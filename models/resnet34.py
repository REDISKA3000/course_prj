class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding = 1,downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels, kernel_size=3,
                               stride=stride, padding=padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels, kernel_size=3,
                               stride=stride,padding=padding)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        x1 = self.relu1(self.batchnorm1(self.conv1(x)))
        x2 = self.relu2(self.batchnorm2(self.conv2(x1)))
        print(x2.shape)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = x2 + identity
        # print(out.shape)
        return out


class ResNet34(nn.Module):

    def layer(self, num_blocks, in_channels, out_channels, downsample=None,
              block=Block, stride=1):
        layers = []
        if in_channels != out_channels or stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          stride=stride, kernel_size=1),
                nn.BatchNorm2d(out_channels))

        layers.append(block(in_channels=in_channels, out_channels=out_channels,
                            downsample=downsample, stride=stride))
        for i in range(num_blocks - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def __init__(self, in_channels=3, out_channels=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                               kernel_size=7, stride=2, padding = 3)
        self.batchnorm = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = self.layer(num_blocks=3, in_channels=64, out_channels=64)
        self.conv3 = self.layer(num_blocks=4, in_channels=64, out_channels=128)
        self.conv4 = self.layer(num_blocks=6, in_channels=128,
                                out_channels=256)
        self.conv5 = self.layer(num_blocks=3, in_channels=256,
                                out_channels=512)

        self.avgpool = nn.AvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=512, out_features=out_channels)
        # self.fc1 = nn.Linear(in_features = out_channels, out_features = 2)

    def feature_extraction(self, x):
        x1 = self.maxpool(self.batchnorm(self.conv1(x)))
        # print(x1.shape)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        return self.avgpool(x5)

    def forward(self, x):
        features = self.feature_extraction(x)

        flat = torch.flatten(features, 1) # torch.flatten(features, 1)

        out = self.fc(flat)
        return out