import torch.nn as nn

# CNN parameters:
input_channels = 1

num_filters = [25, 50, 100, 200]
kernel_temporal = [22, 10, 10, 10]

num_l1 = 12


class DeepConvNet(nn.Module):
    def __init__(self, num_classes, x_dim, y_dim):
        super(DeepConvNet, self).__init__()
        self.num_classes = num_classes
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.temporal_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=num_filters[0],
                      kernel_size=[1, kernel_temporal[0]])
        )

        self.spatial_block1 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters[0],
                      out_channels=num_filters[0],
                      kernel_size=[15, 1]),
            nn.MaxPool2d(kernel_size=[1, 4],
                         stride=[1, 4])
        )

        self.temporal_block2 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters[0],
                      out_channels=num_filters[1],
                      kernel_size=[1, kernel_temporal[1]]),
            nn.BatchNorm2d(num_filters[1]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 4],
                         stride=[1, 4])
        )

        self.temporal_block3 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters[1],
                      out_channels=num_filters[2],
                      kernel_size=[1, kernel_temporal[2]]),
            nn.BatchNorm2d(num_filters[2]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 3],
                         stride=[1, 3])
        )

        self.temporal_block4 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters[2],
                      out_channels=num_filters[3],
                      kernel_size=[1, kernel_temporal[3]]),
            nn.BatchNorm2d(num_filters[3]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 3],
                         stride=[1, 3])
        )

        self.temporal_block5 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters[3],
                      out_channels=num_filters[3],
                      kernel_size=[1, 12]),
            nn.BatchNorm2d(num_filters[3]),
            nn.ELU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(400, num_l1),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(num_l1, num_classes)
        )

    def forward(self, x):
        x = self.temporal_block1(x)  # [, 25, 15, 513]
        x = self.spatial_block1(x)  # [, 25, 1, 171]
        x = self.temporal_block2(x)  # [, 50, 1, 54]
        x = self.temporal_block3(x)  # [, 100, 1, 15]
        x = self.temporal_block4(x)  # [, 200, 1, 2]
        x = self.temporal_block5(x)  # [, 200, 1, 2]
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

