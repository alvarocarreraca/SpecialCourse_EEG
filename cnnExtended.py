import torch
import torch.nn as nn

# CNN parameters:
input_channels = 1
num_filters_conv1 = 8
num_filters_conv2 = 16
num_filters_conv3 = 24
kernel_size_conv1 = 5
kernel_size_conv2 = 5
kernel_size_pooling = 2  # [1, 15]
padding_conv1 = 2
padding_conv2 = 2
num_l1 = 912
num_l2 = 254
num_l3 = 24


class NetExtended(nn.Module):
    def __init__(self, num_classes, x_dim, y_dim):
        super(NetExtended, self).__init__()
        self.num_classes = num_classes
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.features_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=num_filters_conv1,
                      kernel_size=kernel_size_conv1,
                      padding=padding_conv1),
            nn.ELU(),
            nn.BatchNorm2d(num_filters_conv1),
            nn.MaxPool2d(kernel_size=kernel_size_pooling,
                         stride=2)
        )

        self.features_block2 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters_conv1,
                      out_channels=num_filters_conv2,
                      kernel_size=kernel_size_conv2,
                      padding=padding_conv2),
            nn.ELU(),
            nn.BatchNorm2d(num_filters_conv2),
            nn.MaxPool2d(kernel_size=kernel_size_pooling,
                         stride=2)
        )

        self.features_block3 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters_conv2,
                      out_channels=num_filters_conv3,
                      kernel_size=kernel_size_conv2,
                      padding=padding_conv2),
            nn.ELU(),
            nn.BatchNorm2d(num_filters_conv3),
            nn.MaxPool2d(kernel_size=kernel_size_pooling,
                         stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(7680, num_l1),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(num_l1, num_l2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(num_l2, num_l3),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(num_l3, num_classes)
        )

    def forward(self, x):
        x = self.features_block1(x)
        x = self.features_block2(x)
        x = self.features_block3(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

'''
model = NetExtended(2, 3, 640)
X_try = torch.randn(6, 1, 15, 2560)
b = model(X_try)
print(b.shape)
# print(b)
'''
'''
#a = torch.softmax(b,dim=1)
#print(a)

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0, 1, 0, 0, 0, 1])
print(Y - 1)
l1 = loss(b,Y)
print(l1)
'''
