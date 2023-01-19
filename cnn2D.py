import torch.nn as nn

# ---------- CNN parameters:
input_channels = 1

# -------- 7 convolutional blocks:
num_filters = [3, 4, 6, 7, 9, 10, 12]

# ____TEMPORAL: 4 blocks_____
# Convolution parameters:
kernel_temporal = [9, 7, 5, 3]
padding_temporal = [4, 3, 2, 1]  # (kernel_temporal-1)/2
kernel_size_pooling_temp = [1, 2]
pool_stride3_temp = [1, 3]

# SPATIAL: 3 blocks
kernel_spatial = [5, 3, 3]
stride1_spatial = [1, 1]
stride2_spatial = [2, 1]

num_l1 = 46


class Net2D(nn.Module):
    def __init__(self, num_classes, x_dim, y_dim):
        super(Net2D, self).__init__()
        self.num_classes = num_classes
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.temporal_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=num_filters[0],
                      kernel_size=[1, kernel_temporal[0]],
                      padding=[0, padding_temporal[0]]),
            nn.BatchNorm2d(num_filters[0]),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.AvgPool2d(kernel_size=kernel_size_pooling_temp,
                         stride=pool_stride3_temp)
        )

        self.spatial_block1 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters[0],
                      out_channels=num_filters[1],
                      kernel_size=[kernel_spatial[0], 1],
                      stride=stride1_spatial),
            nn.BatchNorm2d(num_filters[1]),
            nn.ELU(),
            nn.Dropout(0.1)
        )

        self.temporal_block2 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters[1],
                      out_channels=num_filters[2],
                      kernel_size=[1, kernel_temporal[1]],
                      padding=[0, padding_temporal[1]]),
            nn.BatchNorm2d(num_filters[2]),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.AvgPool2d(kernel_size=kernel_size_pooling_temp,
                         stride=pool_stride3_temp)
        )

        self.spatial_block2 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters[2],
                      out_channels=num_filters[3],
                      kernel_size=[kernel_spatial[1], 1],
                      stride=stride2_spatial),
            nn.BatchNorm2d(num_filters[3]),
            nn.ELU(),
            nn.Dropout(0.1)
        )

        self.temporal_block3 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters[3],
                      out_channels=num_filters[4],
                      kernel_size=[1, kernel_temporal[2]],
                      padding=[0, padding_temporal[2]]),
            nn.BatchNorm2d(num_filters[4]),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.AvgPool2d(kernel_size=kernel_size_pooling_temp,
                         stride=pool_stride3_temp)
        )

        self.spatial_block3 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters[4],
                      out_channels=num_filters[5],
                      kernel_size=[kernel_spatial[2], 1],
                      stride=stride2_spatial),
            nn.BatchNorm2d(num_filters[5]),
            nn.ELU(),
            nn.Dropout(0.1)
        )

        self.temporal_block4 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters[5],
                      out_channels=num_filters[6],
                      kernel_size=[1, kernel_temporal[3]],
                      padding=[0, padding_temporal[3]]),
            nn.BatchNorm2d(num_filters[6]),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.AvgPool2d(kernel_size=kernel_size_pooling_temp,
                         stride=pool_stride3_temp)
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, num_l1),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(num_l1, num_classes)
        )

    def forward(self, x):
        # input: [, 1, 15, 2560]
        x = self.temporal_block1(x)  # output: [, 3, 15, 854]
        x = self.spatial_block1(x)  # output: [, 4, 11, 854]
        x = self.temporal_block2(x)  # output: [, 6, 11, 284]
        x = self.spatial_block2(x)  # output: [, 7, 5, 284]
        x = self.temporal_block3(x)  # output: [,9, 5, 94]
        x = self.spatial_block3(x)  # output: [, 10, 2, 94]
        x = self.temporal_block4(x)  # output: [, 12, 2, 32]
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
