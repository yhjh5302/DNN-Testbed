import torch
import torch.nn as nn

class AlexNet_features1(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(AlexNet_features1, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features1.load_state_dict(torch.load('./cifar_AlexNet_features1.pth', map_location=torch.device(device)))

    def forward(self, x):
        x = self.features1(x)
        return x

class AlexNet_features2(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(AlexNet_features2, self).__init__()
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features2.load_state_dict(torch.load('./cifar_AlexNet_features2.pth', map_location=torch.device(device)))

    def forward(self, x):
        x = self.features2(x)
        return x

class AlexNet_features3(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(AlexNet_features3, self).__init__()
        self.features3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features3.load_state_dict(torch.load('./cifar_AlexNet_features3.pth', map_location=torch.device(device)))

    def forward(self, x):
        x = self.features3(x)
        return x

class AlexNet_features4(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(AlexNet_features4, self).__init__()
        self.features4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features4.load_state_dict(torch.load('./cifar_AlexNet_features4.pth', map_location=torch.device(device)))

    def forward(self, x):
        x = self.features4(x)
        return x

class AlexNet_features5(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(AlexNet_features5, self).__init__()
        self.features5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Dropout(),
        )
        self.features5.load_state_dict(torch.load('./cifar_AlexNet_features5.pth', map_location=torch.device(device)))

    def forward(self, x):
        x = self.features5(x)
        x = torch.flatten(x, 1)
        return x

class AlexNet_classifier1(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(AlexNet_classifier1, self).__init__()
        self.classifier1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier1.load_state_dict(torch.load('./cifar_AlexNet_classifier1.pth', map_location=torch.device(device)))

    def forward(self, x):
        x = self.classifier1(x)
        return x

class AlexNet_classifier2(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(AlexNet_classifier2, self).__init__()
        self.classifier2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.classifier2.load_state_dict(torch.load('./cifar_AlexNet_classifier2.pth', map_location=torch.device(device)))

    def forward(self, x):
        x = self.classifier2(x)
        return x

class AlexNet_classifier3(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(AlexNet_classifier3, self).__init__()
        self.classifier3 = nn.Sequential(
            nn.Linear(4096, 1000),
        )
        self.classifier3.load_state_dict(torch.load('./cifar_AlexNet_classifier3.pth', map_location=torch.device(device)))

    def forward(self, x):
        x = self.classifier3(x)
        return x