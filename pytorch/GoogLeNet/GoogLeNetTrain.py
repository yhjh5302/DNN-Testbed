import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import math

class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 1000, transform_input: bool = True, init_weights: bool = True) -> None:
        super(GoogLeNet, self).__init__()
        self.transform_input = transform_input
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.conv1_maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
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
        self.conv3_maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
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
        '''
        self.inception_aux1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
        )
        self.inception_aux2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(528, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
        )
        '''
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fully_connected = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._transform_input(x)
        # conv1 and max pool()
        x = self.conv1(x)
        x = self.conv1_maxpool(x)
        # conv2()
        x = self.conv2(x)
        # conv3 and max pool()
        x = self.conv3(x)
        x = self.conv3_maxpool(x)
        # inception3a()
        branch1 = self.inception3a_branch1(x)
        branch2 = self.inception3a_branch2(x)
        branch3 = self.inception3a_branch3(x)
        branch4 = self.inception3a_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        # inception3b and max pool()
        branch1 = self.inception3b_branch1(x)
        branch2 = self.inception3b_branch2(x)
        branch3 = self.inception3b_branch3(x)
        branch4 = self.inception3b_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        x = self.inception3b_maxpool(x)
        # inception4a()
        branch1 = self.inception4a_branch1(x)
        branch2 = self.inception4a_branch2(x)
        branch3 = self.inception4a_branch3(x)
        branch4 = self.inception4a_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        # inception4b()
        branch1 = self.inception4b_branch1(x)
        branch2 = self.inception4b_branch2(x)
        branch3 = self.inception4b_branch3(x)
        branch4 = self.inception4b_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        # inception4c()
        branch1 = self.inception4c_branch1(x)
        branch2 = self.inception4c_branch2(x)
        branch3 = self.inception4c_branch3(x)
        branch4 = self.inception4c_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        # inception4d()
        branch1 = self.inception4d_branch1(x)
        branch2 = self.inception4d_branch2(x)
        branch3 = self.inception4d_branch3(x)
        branch4 = self.inception4d_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        # inception4e()
        branch1 = self.inception4e_branch1(x)
        branch2 = self.inception4e_branch2(x)
        branch3 = self.inception4e_branch3(x)
        branch4 = self.inception4e_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        x = self.inception4e_maxpool(x)
        # inception5a()
        branch1 = self.inception5a_branch1(x)
        branch2 = self.inception5a_branch2(x)
        branch3 = self.inception5a_branch3(x)
        branch4 = self.inception5a_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        # inception5b()
        branch1 = self.inception5b_branch1(x)
        branch2 = self.inception5b_branch2(x)
        branch3 = self.inception5b_branch3(x)
        branch4 = self.inception5b_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        # avg pool, flatten and fully_connected
        x = self.avgpool(x)
        x = self.fully_connected(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))
    
    transform = transforms.Compose([transforms.Resize(size=(224,224),interpolation=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=8)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    epoch_size = 20
    model = GoogLeNet().to(device)

    #from torchsummary import summary
    #summary(model, (3, 224, 224))

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    for epoch in range(epoch_size):   # repeat process with same data
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # receive inputs from data
            inputs, labels = data[0].cuda(), data[1].cuda()
            
            # gradient set to zero
            optimizer.zero_grad()

            # forward and back prop and optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print progress
            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%2d/%2d,%4d/%4d] loss: %.3f' % (epoch + 1, epoch_size, i + 1, len(trainset)/10, running_loss / 1000))
                running_loss = 0.0

    print('Finished Training')

    #### save trained model

    PATH = './'
    torch.save(model.conv1.state_dict(), PATH + 'cifar_GoogLeNet_conv1.pth')
    torch.save(model.conv2.state_dict(), PATH + 'cifar_GoogLeNet_conv2.pth')
    torch.save(model.conv3.state_dict(), PATH + 'cifar_GoogLeNet_conv3.pth')
    torch.save(model.inception3a_branch1.state_dict(), PATH + 'cifar_GoogLeNet_inception3a_branch1.pth')
    torch.save(model.inception3a_branch2.state_dict(), PATH + 'cifar_GoogLeNet_inception3a_branch2.pth')
    torch.save(model.inception3a_branch3.state_dict(), PATH + 'cifar_GoogLeNet_inception3a_branch3.pth')
    torch.save(model.inception3a_branch4.state_dict(), PATH + 'cifar_GoogLeNet_inception3a_branch4.pth')
    torch.save(model.inception3b_branch1.state_dict(), PATH + 'cifar_GoogLeNet_inception3b_branch1.pth')
    torch.save(model.inception3b_branch2.state_dict(), PATH + 'cifar_GoogLeNet_inception3b_branch2.pth')
    torch.save(model.inception3b_branch3.state_dict(), PATH + 'cifar_GoogLeNet_inception3b_branch3.pth')
    torch.save(model.inception3b_branch4.state_dict(), PATH + 'cifar_GoogLeNet_inception3b_branch4.pth')
    torch.save(model.inception4a_branch1.state_dict(), PATH + 'cifar_GoogLeNet_inception4a_branch1.pth')
    torch.save(model.inception4a_branch2.state_dict(), PATH + 'cifar_GoogLeNet_inception4a_branch2.pth')
    torch.save(model.inception4a_branch3.state_dict(), PATH + 'cifar_GoogLeNet_inception4a_branch3.pth')
    torch.save(model.inception4a_branch4.state_dict(), PATH + 'cifar_GoogLeNet_inception4a_branch4.pth')
    torch.save(model.inception4b_branch1.state_dict(), PATH + 'cifar_GoogLeNet_inception4b_branch1.pth')
    torch.save(model.inception4b_branch2.state_dict(), PATH + 'cifar_GoogLeNet_inception4b_branch2.pth')
    torch.save(model.inception4b_branch3.state_dict(), PATH + 'cifar_GoogLeNet_inception4b_branch3.pth')
    torch.save(model.inception4b_branch4.state_dict(), PATH + 'cifar_GoogLeNet_inception4b_branch4.pth')
    torch.save(model.inception4c_branch1.state_dict(), PATH + 'cifar_GoogLeNet_inception4c_branch1.pth')
    torch.save(model.inception4c_branch2.state_dict(), PATH + 'cifar_GoogLeNet_inception4c_branch2.pth')
    torch.save(model.inception4c_branch3.state_dict(), PATH + 'cifar_GoogLeNet_inception4c_branch3.pth')
    torch.save(model.inception4c_branch4.state_dict(), PATH + 'cifar_GoogLeNet_inception4c_branch4.pth')
    torch.save(model.inception4d_branch1.state_dict(), PATH + 'cifar_GoogLeNet_inception4d_branch1.pth')
    torch.save(model.inception4d_branch2.state_dict(), PATH + 'cifar_GoogLeNet_inception4d_branch2.pth')
    torch.save(model.inception4d_branch3.state_dict(), PATH + 'cifar_GoogLeNet_inception4d_branch3.pth')
    torch.save(model.inception4d_branch4.state_dict(), PATH + 'cifar_GoogLeNet_inception4d_branch4.pth')
    torch.save(model.inception4e_branch1.state_dict(), PATH + 'cifar_GoogLeNet_inception4e_branch1.pth')
    torch.save(model.inception4e_branch2.state_dict(), PATH + 'cifar_GoogLeNet_inception4e_branch2.pth')
    torch.save(model.inception4e_branch3.state_dict(), PATH + 'cifar_GoogLeNet_inception4e_branch3.pth')
    torch.save(model.inception4e_branch4.state_dict(), PATH + 'cifar_GoogLeNet_inception4e_branch4.pth')
    torch.save(model.inception5a_branch1.state_dict(), PATH + 'cifar_GoogLeNet_inception5a_branch1.pth')
    torch.save(model.inception5a_branch2.state_dict(), PATH + 'cifar_GoogLeNet_inception5a_branch2.pth')
    torch.save(model.inception5a_branch3.state_dict(), PATH + 'cifar_GoogLeNet_inception5a_branch3.pth')
    torch.save(model.inception5a_branch4.state_dict(), PATH + 'cifar_GoogLeNet_inception5a_branch4.pth')
    torch.save(model.inception5b_branch1.state_dict(), PATH + 'cifar_GoogLeNet_inception5b_branch1.pth')
    torch.save(model.inception5b_branch2.state_dict(), PATH + 'cifar_GoogLeNet_inception5b_branch2.pth')
    torch.save(model.inception5b_branch3.state_dict(), PATH + 'cifar_GoogLeNet_inception5b_branch3.pth')
    torch.save(model.inception5b_branch4.state_dict(), PATH + 'cifar_GoogLeNet_inception5b_branch4.pth')
    #torch.save(model.inception_aux1.state_dict(), PATH + 'cifar_GoogLeNet_inception_aux1.pth')
    #torch.save(model.inception_aux2.state_dict(), PATH + 'cifar_GoogLeNet_inception_aux2.pth')
    torch.save(model.fully_connected.state_dict(), PATH + 'cifar_GoogLeNet_fully_connected.pth')

    print('Model Saved')