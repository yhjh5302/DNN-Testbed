import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

class VGGNet(nn.Module):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True ) -> None:
        super(VGGNet, self).__init__()
        self.features1_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features1_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features2_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features2_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features3_conv1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features3_conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features3_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features4_conv1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features4_conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features4_conv3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features5_conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features5_conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features5_conv3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Dropout(),
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(4096, 1000),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features1_conv1(x)
        x = self.features1_conv2(x)
        x = self.features2_conv1(x)
        x = self.features2_conv2(x)
        x = self.features3_conv1(x)
        x = self.features3_conv2(x)
        x = self.features3_conv3(x)
        x = self.features4_conv1(x)
        x = self.features4_conv2(x)
        x = self.features4_conv3(x)
        x = self.features5_conv1(x)
        x = self.features5_conv2(x)
        x = self.features5_conv3(x)
        x = torch.flatten(x, 1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.classifier3(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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
    model = VGGNet()
    model = model.cuda()
    #from torchsummary import summary
    #summary(model, (3, 224, 224))

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
    torch.save(model.features1_conv1.state_dict(), PATH + 'cifar_VGGNet_features1_conv1.pth')
    torch.save(model.features1_conv2.state_dict(), PATH + 'cifar_VGGNet_features1_conv2.pth')
    torch.save(model.features2_conv1.state_dict(), PATH + 'cifar_VGGNet_features2_conv1.pth')
    torch.save(model.features2_conv2.state_dict(), PATH + 'cifar_VGGNet_features2_conv2.pth')
    torch.save(model.features3_conv1.state_dict(), PATH + 'cifar_VGGNet_features3_conv1.pth')
    torch.save(model.features3_conv2.state_dict(), PATH + 'cifar_VGGNet_features3_conv2.pth')
    torch.save(model.features3_conv3.state_dict(), PATH + 'cifar_VGGNet_features3_conv3.pth')
    torch.save(model.features4_conv1.state_dict(), PATH + 'cifar_VGGNet_features4_conv1.pth')
    torch.save(model.features4_conv2.state_dict(), PATH + 'cifar_VGGNet_features4_conv2.pth')
    torch.save(model.features4_conv3.state_dict(), PATH + 'cifar_VGGNet_features4_conv3.pth')
    torch.save(model.features5_conv1.state_dict(), PATH + 'cifar_VGGNet_features5_conv1.pth')
    torch.save(model.features5_conv2.state_dict(), PATH + 'cifar_VGGNet_features5_conv2.pth')
    torch.save(model.features5_conv3.state_dict(), PATH + 'cifar_VGGNet_features5_conv3.pth')
    torch.save(model.classifier1.state_dict(), PATH + 'cifar_VGGNet_classifier1.pth')
    torch.save(model.classifier2.state_dict(), PATH + 'cifar_VGGNet_classifier2.pth')
    torch.save(model.classifier3.state_dict(), PATH + 'cifar_VGGNet_classifier3.pth')

    print('Model Saved')