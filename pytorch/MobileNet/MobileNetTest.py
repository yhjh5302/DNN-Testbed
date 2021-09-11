import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import time

PATH = './'
num_classes=1000

conv1 = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
)
separable_conv2 = nn.Sequential(
    nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, groups=32, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
)
separable_conv3 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, groups=64, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
)
separable_conv4 = nn.Sequential(
    nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, groups=128, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
)
separable_conv5 = nn.Sequential(
    nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, groups=128, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
)
separable_conv6 = nn.Sequential(
    nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, groups=256, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
)
separable_conv7 = nn.Sequential(
    nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2, groups=256, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
)
separable_conv8 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
)
separable_conv9 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
)
separable_conv10 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
)
separable_conv11 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
)
separable_conv12 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
)
separable_conv13 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2, groups=512, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(inplace=True),
)
separable_conv14 = nn.Sequential(
    nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=1, groups=1024, bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(inplace=True),
    nn.Conv2d(1024, 1024, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d(1),
)
fc = nn.Linear(1024, 1000)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize(size=(224,224),interpolation=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    conv1.load_state_dict(torch.load('./cifar_MobileNetV1_conv1.pth'))
    separable_conv2.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv2.pth'))
    separable_conv3.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv3.pth'))
    separable_conv4.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv4.pth'))
    separable_conv5.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv5.pth'))
    separable_conv6.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv6.pth'))
    separable_conv7.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv7.pth'))
    separable_conv8.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv8.pth'))
    separable_conv9.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv9.pth'))
    separable_conv10.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv10.pth'))
    separable_conv11.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv11.pth'))
    separable_conv12.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv12.pth'))
    separable_conv13.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv13.pth'))
    separable_conv14.load_state_dict(torch.load('./cifar_MobileNetV1_separable_conv14.pth'))
    fc.load_state_dict(torch.load('./cifar_MobileNetV1_fc.pth'))

    correct = 0
    total = 0
    avg_time = 0
    time_conv1 = 0
    time_separable_conv2 = 0
    time_separable_conv3 = 0
    time_separable_conv4 = 0
    time_separable_conv5 = 0
    time_separable_conv6 = 0
    time_separable_conv7 = 0
    time_separable_conv8 = 0
    time_separable_conv9 = 0
    time_separable_conv10 = 0
    time_separable_conv11 = 0
    time_separable_conv12 = 0
    time_separable_conv13 = 0
    time_separable_conv14 = 0
    time_fc = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data[0], data[1]
            
            total_start = time.time()
            
            # conv1
            start = time.time()
            outputs = conv1(images)
            time_conv1 += time.time() - start
            
            # separable_conv2
            start = time.time()
            outputs = separable_conv2(outputs)
            time_separable_conv2 += time.time() - start
            
            # separable_conv3
            start = time.time()
            outputs = separable_conv3(outputs)
            time_separable_conv3 += time.time() - start
            
            # separable_conv4
            start = time.time()
            outputs = separable_conv4(outputs)
            time_separable_conv4 += time.time() - start
            
            # separable_conv5
            start = time.time()
            outputs = separable_conv5(outputs)
            time_separable_conv5 += time.time() - start
            
            # separable_conv6
            start = time.time()
            outputs = separable_conv6(outputs)
            time_separable_conv6 += time.time() - start
            
            # separable_conv7
            start = time.time()
            outputs = separable_conv7(outputs)
            time_separable_conv7 += time.time() - start
            
            # separable_conv8
            start = time.time()
            outputs = separable_conv8(outputs)
            time_separable_conv8 += time.time() - start
            
            # separable_conv9
            start = time.time()
            outputs = separable_conv9(outputs)
            time_separable_conv9 += time.time() - start
            
            # separable_conv10
            start = time.time()
            outputs = separable_conv10(outputs)
            time_separable_conv10 += time.time() - start
            
            # separable_conv11
            start = time.time()
            outputs = separable_conv11(outputs)
            time_separable_conv11 += time.time() - start
            
            # separable_conv12
            start = time.time()
            outputs = separable_conv12(outputs)
            time_separable_conv12 += time.time() - start
            
            # separable_conv13
            start = time.time()
            outputs = separable_conv13(outputs)
            time_separable_conv13 += time.time() - start
            
            # separable_conv14
            start = time.time()
            outputs = separable_conv14(outputs)
            time_separable_conv14 += time.time() - start
            
            outputs = outputs.view(outputs.size(0), -1)
            
            # fc
            start = time.time()
            outputs = fc(outputs)
            time_fc += time.time() - start
            
            _, predicted = torch.max(outputs.data, 1)
            avg_time += time.time() - total_start
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i == 99:
                print('total: ', total)
                print('correct: ', correct)
                print('avg_time: ', avg_time / 100)
                print("time_conv1: ", time_conv1 / 100)
                print("time_separable_conv2: ", time_separable_conv2 / 100)
                print("time_separable_conv3: ", time_separable_conv3 / 100)
                print("time_separable_conv4: ", time_separable_conv4 / 100)
                print("time_separable_conv5: ", time_separable_conv5 / 100)
                print("time_separable_conv6: ", time_separable_conv6 / 100)
                print("time_separable_conv7: ", time_separable_conv7 / 100)
                print("time_separable_conv8: ", time_separable_conv8 / 100)
                print("time_separable_conv9: ", time_separable_conv9 / 100)
                print("time_separable_conv10: ", time_separable_conv10 / 100)
                print("time_separable_conv11: ", time_separable_conv11 / 100)
                print("time_separable_conv12: ", time_separable_conv12 / 100)
                print("time_separable_conv13: ", time_separable_conv13 / 100)
                print("time_separable_conv14: ", time_separable_conv14 / 100)
                print("time_fc: ", time_fc / 100)
                break
            if i % 100 == 99:
                print('[%3d/%3d] tested' % (i + 1, len(testset) / 10))

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))