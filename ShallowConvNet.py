import torch.nn as nn

# CNN parameters:
input_channels = 1
channel_temp = 40
num_l1 = 12


class ShallowNet(nn.Module):
    def __init__(self, num_classes, x_dim, y_dim):
        super(ShallowNet, self).__init__()
        self.num_classes = num_classes
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.temp_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=channel_temp,
                      kernel_size=[1, 61])
        )

        self.spac_conv = nn.Sequential(
            nn.Conv2d(in_channels=channel_temp,
                      out_channels=channel_temp,
                      kernel_size=[15, 1])
        )

        self.pool_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=[1, 100],
                         stride=[1, 100])
        )

        self.classifier = nn.Sequential(
            nn.Linear(1000, num_l1),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(num_l1, num_classes)
        )

    def forward(self, x):
        x = self.temp_conv(x)  # (,40,15,2500)
        x = self.spac_conv(x)  # (,40,1,2500)
        x = self.pool_block(x)  # (,40,1,25)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
