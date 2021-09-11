import torch
import torch.nn as nn

class GoogLeNet_conv(nn.Module):
    def __init__(self, num_classes=1000, device=torch.device('cpu')):
        super(GoogLeNet_conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        conv1_maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(192, eps=0.001),
            nn.ReLU(inplace=True),
        )
        conv3_maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv1.load_state_dict(torch.load('./cifar_GoogLeNet_conv1.pth', map_location=torch.device(device)))
        self.conv2.load_state_dict(torch.load('./cifar_GoogLeNet_conv2.pth', map_location=torch.device(device)))
        self.conv3.load_state_dict(torch.load('./cifar_GoogLeNet_conv3.pth', map_location=torch.device(device)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_maxpool(x)
        return x

class GoogLeNet_inception3a(nn.Module):
    def __init__(self, num_classes=1000, device=torch.device('cpu')):
        super(GoogLeNet_inception3a, self).__init__()
        self.inception3a_branch1 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3a_branch2 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(96, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3a_branch3 = nn.Sequential(
            nn.Conv2d(192, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3a_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(192, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3a_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception3a_branch1.pth', map_location=torch.device(device)))
        self.inception3a_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception3a_branch2.pth', map_location=torch.device(device)))
        self.inception3a_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception3a_branch3.pth', map_location=torch.device(device)))
        self.inception3a_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception3a_branch4.pth', map_location=torch.device(device)))

    def forward(self, x):
        branch1 = self.inception3a_branch1(x)
        branch2 = self.inception3a_branch2(x)
        branch3 = self.inception3a_branch3(x)
        branch4 = self.inception3a_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        return x

class GoogLeNet_inception3b(nn.Module):
    def __init__(self, num_classes=1000, device=torch.device('cpu')):
        super(GoogLeNet_inception3b, self).__init__()
        self.inception3b_branch1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3b_branch2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(192, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3b_branch3 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3b_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3b_maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception3b_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception3b_branch1.pth', map_location=torch.device(device)))
        self.inception3b_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception3b_branch2.pth', map_location=torch.device(device)))
        self.inception3b_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception3b_branch3.pth', map_location=torch.device(device)))
        self.inception3b_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception3b_branch4.pth', map_location=torch.device(device)))

    def forward(self, x):
        branch1 = self.inception3b_branch1(x)
        branch2 = self.inception3b_branch2(x)
        branch3 = self.inception3b_branch3(x)
        branch4 = self.inception3b_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        x = self.inception3b_maxpool(x)
        return x

class GoogLeNet_inception4a(nn.Module):
    def __init__(self, num_classes=1000, device=torch.device('cpu')):
        super(GoogLeNet_inception4a, self).__init__()
        self.inception4a_branch1 = nn.Sequential(
            nn.Conv2d(480, 192, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(192, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4a_branch2 = nn.Sequential(
            nn.Conv2d(480, 96, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(96, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 208, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(208, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4a_branch3 = nn.Sequential(
            nn.Conv2d(480, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4a_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(480, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4a_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception4a_branch1.pth', map_location=torch.device(device)))
        self.inception4a_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception4a_branch2.pth', map_location=torch.device(device)))
        self.inception4a_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception4a_branch3.pth', map_location=torch.device(device)))
        self.inception4a_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception4a_branch4.pth', map_location=torch.device(device)))

    def forward(self, x):
        branch1 = self.inception4a_branch1(x)
        branch2 = self.inception4a_branch2(x)
        branch3 = self.inception4a_branch3(x)
        branch4 = self.inception4a_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        return x

class GoogLeNet_inception4b(nn.Module):
    def __init__(self, num_classes=1000, device=torch.device('cpu')):
        super(GoogLeNet_inception4b, self).__init__()
        self.inception4b_branch1 = nn.Sequential(
            nn.Conv2d(512, 160, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(160, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4b_branch2 = nn.Sequential(
            nn.Conv2d(512, 112, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(112, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(112, 224, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(224, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4b_branch3 = nn.Sequential(
            nn.Conv2d(512, 24, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(24, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4b_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4b_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception4b_branch1.pth', map_location=torch.device(device)))
        self.inception4b_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception4b_branch2.pth', map_location=torch.device(device)))
        self.inception4b_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception4b_branch3.pth', map_location=torch.device(device)))
        self.inception4b_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception4b_branch4.pth', map_location=torch.device(device)))

    def forward(self, x):
        branch1 = self.inception4b_branch1(x)
        branch2 = self.inception4b_branch2(x)
        branch3 = self.inception4b_branch3(x)
        branch4 = self.inception4b_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        return x

class GoogLeNet_inception4c(nn.Module):
    def __init__(self, num_classes=1000, device=torch.device('cpu')):
        super(GoogLeNet_inception4c, self).__init__()
        self.inception4c_branch1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4c_branch2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4c_branch3 = nn.Sequential(
            nn.Conv2d(512, 24, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(24, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4c_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4c_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception4c_branch1.pth', map_location=torch.device(device)))
        self.inception4c_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception4c_branch2.pth', map_location=torch.device(device)))
        self.inception4c_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception4c_branch3.pth', map_location=torch.device(device)))
        self.inception4c_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception4c_branch4.pth', map_location=torch.device(device)))

    def forward(self, x):
        branch1 = self.inception4c_branch1(x)
        branch2 = self.inception4c_branch2(x)
        branch3 = self.inception4c_branch3(x)
        branch4 = self.inception4c_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        return x

class GoogLeNet_inception4d(nn.Module):
    def __init__(self, num_classes=1000, device=torch.device('cpu')):
        super(GoogLeNet_inception4d, self).__init__()
        self.inception4d_branch1 = nn.Sequential(
            nn.Conv2d(512, 112, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(112, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4d_branch2 = nn.Sequential(
            nn.Conv2d(512, 144, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(144, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(144, 288, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(288, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4d_branch3 = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4d_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4d_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception4d_branch1.pth', map_location=torch.device(device)))
        self.inception4d_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception4d_branch2.pth', map_location=torch.device(device)))
        self.inception4d_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception4d_branch3.pth', map_location=torch.device(device)))
        self.inception4d_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception4d_branch4.pth', map_location=torch.device(device)))

    def forward(self, x):
        branch1 = self.inception4d_branch1(x)
        branch2 = self.inception4d_branch2(x)
        branch3 = self.inception4d_branch3(x)
        branch4 = self.inception4d_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        return x

class GoogLeNet_inception4e(nn.Module):
    def __init__(self, num_classes=1000, device=torch.device('cpu')):
        super(GoogLeNet_inception4e, self).__init__()
        self.inception4e_branch1 = nn.Sequential(
            nn.Conv2d(528, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4e_branch2 = nn.Sequential(
            nn.Conv2d(528, 160, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(160, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(320, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4e_branch3 = nn.Sequential(
            nn.Conv2d(528, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4e_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(528, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4e_maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.inception4e_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception4e_branch1.pth', map_location=torch.device(device)))
        self.inception4e_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception4e_branch2.pth', map_location=torch.device(device)))
        self.inception4e_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception4e_branch3.pth', map_location=torch.device(device)))
        self.inception4e_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception4e_branch4.pth', map_location=torch.device(device)))

    def forward(self, x):
        branch1 = self.inception4e_branch1(x)
        branch2 = self.inception4e_branch2(x)
        branch3 = self.inception4e_branch3(x)
        branch4 = self.inception4e_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        x = self.inception4e_maxpool(x)
        return x

class GoogLeNet_inception5a(nn.Module):
    def __init__(self, num_classes=1000, device=torch.device('cpu')):
        super(GoogLeNet_inception5a, self).__init__()
        self.inception5a_branch1 = nn.Sequential(
            nn.Conv2d(832, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5a_branch2 = nn.Sequential(
            nn.Conv2d(832, 160, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(160, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(320, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5a_branch3 = nn.Sequential(
            nn.Conv2d(832, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5a_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(832, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5a_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception5a_branch1.pth', map_location=torch.device(device)))
        self.inception5a_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception5a_branch2.pth', map_location=torch.device(device)))
        self.inception5a_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception5a_branch3.pth', map_location=torch.device(device)))
        self.inception5a_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception5a_branch4.pth', map_location=torch.device(device)))

    def forward(self, x):
        branch1 = self.inception5a_branch1(x)
        branch2 = self.inception5a_branch2(x)
        branch3 = self.inception5a_branch3(x)
        branch4 = self.inception5a_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        return x

class GoogLeNet_inception5b(nn.Module):
    def __init__(self, num_classes=1000, device=torch.device('cpu')):
        super(GoogLeNet_inception5b, self).__init__()
        self.inception5b_branch1 = nn.Sequential(
            nn.Conv2d(832, 384, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(384, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5b_branch2 = nn.Sequential(
            nn.Conv2d(832, 192, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(192, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5b_branch3 = nn.Sequential(
            nn.Conv2d(832, 48, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(48, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5b_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(832, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5b_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception5b_branch1.pth', map_location=torch.device(device)))
        self.inception5b_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception5b_branch2.pth', map_location=torch.device(device)))
        self.inception5b_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception5b_branch3.pth', map_location=torch.device(device)))
        self.inception5b_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception5b_branch4.pth', map_location=torch.device(device)))

    def forward(self, x):
        branch1 = self.inception5b_branch1(x)
        branch2 = self.inception5b_branch2(x)
        branch3 = self.inception5b_branch3(x)
        branch4 = self.inception5b_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        x = self.avgpool(x)
        return x

class GoogLeNet_fully_connected(nn.Module):
    def __init__(self, num_classes=1000, device=torch.device('cpu')):
        super(GoogLeNet_fully_connected, self).__init__()
        self.fully_connected = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(0.2),
            nn.Linear(1024, 1000),
        )
        self.fully_connected.load_state_dict(torch.load('./cifar_GoogLeNet_fully_connected.pth', map_location=torch.device(device)))

    def forward(self, x):
        x = self.fully_connected(x)
        return x