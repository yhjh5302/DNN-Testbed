import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import math

class MobileNetV1(nn.Module):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True ) -> None:
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.separable_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.separable_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.separable_conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.separable_conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.separable_conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.separable_conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.separable_conv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.separable_conv9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.separable_conv10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.separable_conv11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.separable_conv12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.separable_conv13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.separable_conv14 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=1, groups=1024, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(1024, 1000)
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.separable_conv2(x)
        x = self.separable_conv3(x)
        x = self.separable_conv4(x)
        x = self.separable_conv5(x)
        x = self.separable_conv6(x)
        x = self.separable_conv7(x)
        x = self.separable_conv8(x)
        x = self.separable_conv9(x)
        x = self.separable_conv10(x)
        x = self.separable_conv11(x)
        x = self.separable_conv12(x)
        x = self.separable_conv13(x)
        x = self.separable_conv14(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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
    model = MobileNetV1()
    model = model.cuda()
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
    torch.save(model.conv1.state_dict(), PATH + 'cifar_MobileNetV1_conv1.pth')
    torch.save(model.separable_conv2.state_dict(), PATH + 'cifar_MobileNetV1_separable_conv2.pth')
    torch.save(model.separable_conv3.state_dict(), PATH + 'cifar_MobileNetV1_separable_conv3.pth')
    torch.save(model.separable_conv4.state_dict(), PATH + 'cifar_MobileNetV1_separable_conv4.pth')
    torch.save(model.separable_conv5.state_dict(), PATH + 'cifar_MobileNetV1_separable_conv5.pth')
    torch.save(model.separable_conv6.state_dict(), PATH + 'cifar_MobileNetV1_separable_conv6.pth')
    torch.save(model.separable_conv7.state_dict(), PATH + 'cifar_MobileNetV1_separable_conv7.pth')
    torch.save(model.separable_conv8.state_dict(), PATH + 'cifar_MobileNetV1_separable_conv8.pth')
    torch.save(model.separable_conv9.state_dict(), PATH + 'cifar_MobileNetV1_separable_conv9.pth')
    torch.save(model.separable_conv10.state_dict(), PATH + 'cifar_MobileNetV1_separable_conv10.pth')
    torch.save(model.separable_conv11.state_dict(), PATH + 'cifar_MobileNetV1_separable_conv11.pth')
    torch.save(model.separable_conv12.state_dict(), PATH + 'cifar_MobileNetV1_separable_conv12.pth')
    torch.save(model.separable_conv13.state_dict(), PATH + 'cifar_MobileNetV1_separable_conv13.pth')
    torch.save(model.separable_conv14.state_dict(), PATH + 'cifar_MobileNetV1_separable_conv14.pth')
    torch.save(model.fc.state_dict(), PATH + 'cifar_MobileNetV1_fc.pth')

    print('Model Saved')