import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import math, time

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
    torch.cuda.set_per_process_memory_fraction(fraction=0.2, device=device)
    print(device)
    print(torch.cuda.get_device_name(0))
    half = True

    transform = transforms.Compose([transforms.Resize(size=(224,224),interpolation=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    if half == True:
        model = MobileNetV1().half().cuda()
    else:
        model = MobileNetV1().cuda()

    model.conv1.load_state_dict(torch.load('./cifar_MobileNetV1_conv1.pth'))
    model.separable_conv2.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv2.pth'))
    model.separable_conv3.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv3.pth'))
    model.separable_conv4.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv4.pth'))
    model.separable_conv5.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv5.pth'))
    model.separable_conv6.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv6.pth'))
    model.separable_conv7.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv7.pth'))
    model.separable_conv8.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv8.pth'))
    model.separable_conv9.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv9.pth'))
    model.separable_conv10.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv10.pth'))
    model.separable_conv11.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv11.pth'))
    model.separable_conv12.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv12.pth'))
    model.separable_conv13.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv13.pth'))
    model.separable_conv14.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv14.pth'))
    model.fc.load_state_dict(torch.load('./cifar_MobileNetV1_fc.pth'))

    correct = 0
    total = 0
    avg_time = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data[0], data[1]
            if half == True:
                images = images.to(torch.float16)
            
            total_start = time.time()
            
            # MobileNet FP32
            outputs = model(images.cuda())
            
            _, predicted = torch.max(outputs.data, 1)
            avg_time += time.time() - total_start
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()
            if i == 99:
                print('total: ', total)
                print('correct: ', correct)
                print('avg_time: ', avg_time / 100)
                break
            # if i % 100 == 99:
            #     print('[%3d/%3d] tested' % (i + 1, len(testset) / 10))

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))