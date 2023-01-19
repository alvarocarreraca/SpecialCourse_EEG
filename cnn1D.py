import torch.nn as nn

# CNN parameters:
input_channels = 15
num_filters_conv1 = 12
num_filters_conv2 = 8
num_filters_conv3 = 4
num_filters_conv4 = 2
kernel_size_conv1 = 40
kernel_size_conv2 = 11
kernel_size_conv3 = 7
kernel_size_conv4 = 3
kernel_size_pooling = 2
padding_conv1 = 19
padding_conv2 = 5
padding_conv3 = 3
padding_conv4 = 2
pool_stride1 = 10
pool_stride2 = 4
pool_stride3 = 4
num_l1 = 4
dOut = 0.1


class NetExtended1D(nn.Module):
    def __init__(self, num_classes, x_dim, y_dim):
        super(NetExtended1D, self).__init__()
        self.num_classes = num_classes
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.features_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels,
                      out_channels=num_filters_conv1,
                      kernel_size=kernel_size_conv1,
                      padding=padding_conv1),
            nn.BatchNorm1d(num_filters_conv1),
            nn.ELU(),
            nn.Dropout(dOut),
            nn.AvgPool1d(kernel_size=kernel_size_pooling,
                         stride=pool_stride1)
        )

        self.features_block2 = nn.Sequential(
            nn.Conv1d(in_channels=num_filters_conv1,
                      out_channels=num_filters_conv2,
                      kernel_size=kernel_size_conv2,
                      padding=padding_conv2),
            nn.BatchNorm1d(num_filters_conv2),
            nn.ELU(),
            nn.Dropout(dOut),
            nn.AvgPool1d(kernel_size=kernel_size_pooling,
                         stride=pool_stride2)
        )

        self.features_block3 = nn.Sequential(
            nn.Conv1d(in_channels=num_filters_conv2,
                      out_channels=num_filters_conv3,
                      kernel_size=kernel_size_conv3,
                      padding=padding_conv3),
            nn.BatchNorm1d(num_filters_conv3),
            nn.ELU(),
            nn.Dropout(dOut),
            nn.AvgPool1d(kernel_size=kernel_size_pooling,
                         stride=pool_stride3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, num_l1),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(num_l1, num_classes)
        )

    def forward(self, x):
        x = self.features_block1(x)
        x = self.features_block2(x)
        x = self.features_block3(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x


