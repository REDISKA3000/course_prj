class UNet_FC(nn.Module):

    def __init__(self, in_features):
        super().__init__()

        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

        self.fc0 = nn.Linear(in_features=in_features, out_features=in_features)

        # encoder
        self.fc1 = nn.Linear(in_features=in_features, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=128)
        self.fc5 = nn.Linear(in_features=128, out_features=8)

        # decoder
        self.fc6 = nn.Linear(in_features=8, out_features=128)
        self.fc7 = nn.Linear(in_features=128 * 2, out_features=128)
        self.fc8 = nn.Linear(in_features=128 * 2, out_features=128)
        self.fc9 = nn.Linear(in_features=128 * 2, out_features=128)

        self.out = nn.Linear(in_features=128 * 2, out_features=in_features)

    def forward(self, x):
        input = self.fc0(x)

        x1 = self.relu(self.bn(self.fc1(input)))
        x2 = self.relu(self.bn(self.fc2(x1)))
        x3 = self.relu(self.bn(self.fc3(x2)))
        x4 = self.relu(self.bn(self.fc4(x3)))
        x5 = self.relu(self.fc5(x4))

        xy = [x5, x4, x3, x2, x1]

        x6 = self.relu(self.fc6(xy[0]))
        con1 = torch.cat((x6, xy[1]), 1)
        x7 = self.relu(self.bn(self.fc7(con1)))
        con2 = torch.cat((x7, xy[2]), 1)
        x8 = self.relu(self.bn(self.fc8(con2)))
        con3 = torch.cat((x8, xy[3]), 1)
        x9 = self.relu(self.bn(self.fc9(con3)))
        con4 = torch.cat((x9, xy[4]), 1)

        x10 = self.out(con4)

        # return decoded
        return x10