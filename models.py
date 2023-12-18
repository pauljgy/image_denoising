# Establish neural network model
import torch.nn as nn
import torch

IMAGE_DIM = 64

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.actv1 = nn.ReLU()

    def forward(self, x):
        x = self.actv1(self.conv1(x))
        return x

class DnCNN(nn.Module):
  def __init__(self, depth = 5):
    super().__init__()

    net = torch.nn.ModuleList([])

    net.append(torch.nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size= 3, padding = 1, bias = True))
    net.append(torch.nn.ReLU())

    for i in range(depth):
      net.append(torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, padding = 1, bias = False))
      net.append(torch.nn.BatchNorm2d(num_features = 64, eps = 0.0001, momentum = 0.95))
      net.append(torch.nn.ReLU())

    net.append(torch.nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size= 3, padding = 1, bias = False))
    net.append(torch.nn.Flatten())
    net.append(torch.nn.Linear(IMAGE_DIM**2, IMAGE_DIM**2))

    self.model = torch.nn.Sequential(*net)

  def forward(self, x):

    # batch size
    batch = len(x)

    temp = self.model(x)

    return torch.reshape(temp, (batch, 1, IMAGE_DIM, IMAGE_DIM))
  
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        #1 if batched, 0 if unbatched
        self.cat_dim = 1

        self.conv1_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv1_2 = ConvBlock(in_channels=64, out_channels=64)

        self.maxPool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = ConvBlock(in_channels=64, out_channels=128)
        self.conv2_2 = ConvBlock(in_channels=128, out_channels=128)

        self.convTranspose1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        #DONT FORGET TO ADD CONCATENATION CHANNELS
        self.conv3_1 = ConvBlock(in_channels=128, out_channels=64)
        self.conv3_2 = ConvBlock(in_channels=64, out_channels=1)

    def forward(self, x):
        x = self.conv1_2(self.conv1_1(x))

        skip_connect = x

        x = self.maxPool1(x)

        x = self.conv2_2(self.conv2_1(x))

        x = self.convTranspose1(x)

        x = torch.cat([skip_connect, x], self.cat_dim)

        x = self.conv3_2(self.conv3_1(x))

        return x

class Twonet(nn.Module):
    def __init__(self):
        super(Twonet, self).__init__()
        
        #1 if batched, 0 if unbatched
        self.cat_dimension = 1
        
        self.conv1_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv1_2 = ConvBlock(in_channels=64, out_channels=64)

        self.maxPool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = ConvBlock(in_channels=64, out_channels=128)
        self.conv2_2 = ConvBlock(in_channels=128, out_channels=128)
        
        self.maxPool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = ConvBlock(in_channels=128, out_channels=256)
        self.conv3_2 = ConvBlock(in_channels=256, out_channels=256)
        
        self.convTranspose1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        
        self.conv4_1 = ConvBlock(in_channels=256, out_channels=128)
        self.conv4_2 = ConvBlock(in_channels=128, out_channels=128)

        self.convTranspose2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        
        self.conv5_1 = ConvBlock(in_channels=128, out_channels=64)
        self.conv5_2 = ConvBlock(in_channels=64, out_channels=1)

        #DONT FORGET TO ADD CONCATENATION CHANNELS

    def forward(self, x):
        x = self.conv1_2(self.conv1_1(x))

        skip_connect1 = x

        x = self.maxPool1(x)
        x = self.conv2_2(self.conv2_1(x))
        
        skip_connect2 = x
        
        x = self.maxPool2(x)
        x = self.conv3_2(self.conv3_1(x))

        x = self.convTranspose1(x)

        x = torch.cat([skip_connect2, x], self.cat_dimension)
        x = self.conv4_2(self.conv4_1(x))
        
        x = self.convTranspose2(x)
        
        x = torch.cat([skip_connect1, x], self.cat_dimension)
        x = self.conv5_2(self.conv5_1(x))

        return x

class Threenet(nn.Module):
    def __init__(self):
        super(self).__init__()
        
        #1 if batched, 0 if unbatched
        self.cat_dimension = 1
        
        self.conv1_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv1_2 = ConvBlock(in_channels=64, out_channels=64)

        self.maxPool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = ConvBlock(in_channels=64, out_channels=128)
        self.conv2_2 = ConvBlock(in_channels=128, out_channels=128)
        
        self.maxPool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = ConvBlock(in_channels=128, out_channels=256)
        self.conv3_2 = ConvBlock(in_channels=256, out_channels=256)

        self.maxPool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = ConvBlock(in_channels=256, out_channels=512)
        self.conv4_2 = ConvBlock(in_channels=512, out_channels=512)
        
        self.convTranspose1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        
        self.conv5_1 = ConvBlock(in_channels=512, out_channels=256)
        self.conv5_2 = ConvBlock(in_channels=256, out_channels=256)

        self.convTranspose2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        
        self.conv6_1 = ConvBlock(in_channels=256, out_channels=128)
        self.conv6_2 = ConvBlock(in_channels=128, out_channels=128)

        self.convTranspose3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.conv7_1 = ConvBlock(in_channels=128, out_channels=64)
        self.conv7_2 = ConvBlock(in_channels=64, out_channels=64)
        self.conv7_3 = ConvBlock(in_channels=64, out_channels=1)

        #DONT FORGET TO ADD CONCATENATION CHANNELS

    def forward(self, x):
        x = self.conv1_2(self.conv1_1(x))

        skip_connect1 = x

        x = self.maxPool1(x)
        x = self.conv2_2(self.conv2_1(x))
        
        skip_connect2 = x
        
        x = self.maxPool2(x)
        x = self.conv3_2(self.conv3_1(x))

        skip_connect3 = x

        x = self.maxPool3(x)
        x = self.conv4_2(self.conv4_1(x))

        x = self.convTranspose1(x)
        x = torch.cat([skip_connect3, x], self.cat_dimension)
        x = self.conv5_2(self.conv5_1(x))
        
        x = self.convTranspose2(x)
        x = torch.cat([skip_connect2, x], self.cat_dimension)
        x = self.conv6_2(self.conv6_1(x))

        x = self.convTranspose3(x)
        x = torch.cat([skip_connect1, x], self.cat_dimension)
        x = self.conv7_3(self.conv7_2(self.conv7_1(x)))

        return x