import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN parameters:
input_channels = 1
num_filters_conv1 = 8
num_filters_conv2 = 16
kernel_size_conv1 = 5
kernel_size_conv2 = 5
kernel_size_pooling = 2
padding_conv1 = 2
padding_conv2 = 2
num_l1 = 1200
num_l2 = 840
num_l3 = 64


class Net(nn.Module):
    def __init__(self, num_classes, x_dim, y_dim):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=num_filters_conv1,
                               kernel_size=kernel_size_conv1,
                               padding=padding_conv1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling,
                                 stride=2)
        self.conv2 = nn.Conv2d(in_channels=num_filters_conv1,
                               out_channels=num_filters_conv2,
                               kernel_size=kernel_size_conv2,
                               padding=padding_conv2)

        # set the fully connected layer:
        self.fc1 = nn.Linear(num_filters_conv2*x_dim*y_dim, num_l1)
        self.fc2 = nn.Linear(num_l1, num_l2)
        self.fc3 = nn.Linear(num_l2, num_l3)
        self.fc4 = nn.Linear(num_l3, num_classes)

    def forward(self, x):
        x_cl1 = self.pool(F.relu(self.conv1(x)))
        x_cl2 = self.pool(F.relu(self.conv2(x_cl1)))
        x_flattened = x_cl2.view(-1, num_filters_conv2*self.x_dim*self.y_dim)
        x_fc1 = F.relu(self.fc1(x_flattened))
        x_fc2 = F.relu(self.fc2(x_fc1))
        x_fc3 = F.relu(self.fc3(x_fc2))
        x = self.fc4(x_fc3)  # the softmax activation function is already included in 'CrossEntropyLoss'

        return x


'''
model = Net(2, 3, 640)
x = torch.randn(6, 1, 15, 2560)
b = model(x)
print(b.shape)
print(b)
#a = torch.softmax(b,dim=1)
#print(a)

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0, 1, 0, 0, 0, 1])
print(Y - 1)
l1 = loss(b,Y)
print(l1)
'''
