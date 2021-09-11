import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import time

PATH = './'
num_classes=1000

conv1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
    nn.BatchNorm2d(64, eps=0.001),
    nn.ReLU(inplace=True),
)
conv1_maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
conv2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(64, eps=0.001),
    nn.ReLU(inplace=True),
)
conv3 = nn.Sequential(
    nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(192, eps=0.001),
    nn.ReLU(inplace=True),
)
conv3_maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
inception3a_branch1 = nn.Sequential(
    nn.Conv2d(192, 64, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(64, eps=0.001),
    nn.ReLU(inplace=True),
)
inception3a_branch2 = nn.Sequential(
    nn.Conv2d(192, 96, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(96, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(128, eps=0.001),
    nn.ReLU(inplace=True),
)
inception3a_branch3 = nn.Sequential(
    nn.Conv2d(192, 16, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(16, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(32, eps=0.001),
    nn.ReLU(inplace=True),
)
inception3a_branch4 = nn.Sequential(
    nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
    nn.Conv2d(192, 32, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(32, eps=0.001),
    nn.ReLU(inplace=True),
)
inception3b_branch1 = nn.Sequential(
    nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(128, eps=0.001),
    nn.ReLU(inplace=True),
)
inception3b_branch2 = nn.Sequential(
    nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(128, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(192, eps=0.001),
    nn.ReLU(inplace=True),
)
inception3b_branch3 = nn.Sequential(
    nn.Conv2d(256, 32, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(32, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(96, eps=0.001),
    nn.ReLU(inplace=True),
)
inception3b_branch4 = nn.Sequential(
    nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
    nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(64, eps=0.001),
    nn.ReLU(inplace=True),
)
inception3b_maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
inception4a_branch1 = nn.Sequential(
    nn.Conv2d(480, 192, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(192, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4a_branch2 = nn.Sequential(
    nn.Conv2d(480, 96, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(96, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(96, 208, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(208, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4a_branch3 = nn.Sequential(
    nn.Conv2d(480, 16, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(16, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(16, 48, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(48, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4a_branch4 = nn.Sequential(
    nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
    nn.Conv2d(480, 64, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(64, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4b_branch1 = nn.Sequential(
    nn.Conv2d(512, 160, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(160, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4b_branch2 = nn.Sequential(
    nn.Conv2d(512, 112, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(112, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(112, 224, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(224, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4b_branch3 = nn.Sequential(
    nn.Conv2d(512, 24, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(24, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(64, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4b_branch4 = nn.Sequential(
    nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
    nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(64, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4c_branch1 = nn.Sequential(
    nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(128, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4c_branch2 = nn.Sequential(
    nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(128, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(256, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4c_branch3 = nn.Sequential(
    nn.Conv2d(512, 24, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(24, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(64, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4c_branch4 = nn.Sequential(
    nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
    nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(64, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4d_branch1 = nn.Sequential(
    nn.Conv2d(512, 112, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(112, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4d_branch2 = nn.Sequential(
    nn.Conv2d(512, 144, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(144, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(144, 288, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(288, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4d_branch3 = nn.Sequential(
    nn.Conv2d(512, 32, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(32, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(64, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4d_branch4 = nn.Sequential(
    nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
    nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(64, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4e_branch1 = nn.Sequential(
    nn.Conv2d(528, 256, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(256, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4e_branch2 = nn.Sequential(
    nn.Conv2d(528, 160, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(160, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(320, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4e_branch3 = nn.Sequential(
    nn.Conv2d(528, 32, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(32, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(128, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4e_branch4 = nn.Sequential(
    nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
    nn.Conv2d(528, 128, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(128, eps=0.001),
    nn.ReLU(inplace=True),
)
inception4e_maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
inception5a_branch1 = nn.Sequential(
    nn.Conv2d(832, 256, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(256, eps=0.001),
    nn.ReLU(inplace=True),
)
inception5a_branch2 = nn.Sequential(
    nn.Conv2d(832, 160, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(160, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(320, eps=0.001),
    nn.ReLU(inplace=True),
)
inception5a_branch3 = nn.Sequential(
    nn.Conv2d(832, 32, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(32, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(128, eps=0.001),
    nn.ReLU(inplace=True),
)
inception5a_branch4 = nn.Sequential(
    nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
    nn.Conv2d(832, 128, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(128, eps=0.001),
    nn.ReLU(inplace=True),
)
inception5b_branch1 = nn.Sequential(
    nn.Conv2d(832, 384, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(384, eps=0.001),
    nn.ReLU(inplace=True),
)
inception5b_branch2 = nn.Sequential(
    nn.Conv2d(832, 192, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(192, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(384, eps=0.001),
    nn.ReLU(inplace=True),
)
inception5b_branch3 = nn.Sequential(
    nn.Conv2d(832, 48, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(48, eps=0.001),
    nn.ReLU(inplace=True),
    nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(128, eps=0.001),
    nn.ReLU(inplace=True),
)
inception5b_branch4 = nn.Sequential(
    nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
    nn.Conv2d(832, 128, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(128, eps=0.001),
    nn.ReLU(inplace=True),
)
'''
inception_aux1 = nn.Sequential(
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
inception_aux2 = nn.Sequential(
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
avgpool = nn.AdaptiveAvgPool2d((1, 1))
fully_connected = nn.Sequential(
    nn.Flatten(start_dim=1),
    nn.Dropout(0.2),
    nn.Linear(1024, num_classes),
)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize(size=(224,224),interpolation=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    conv1.load_state_dict(torch.load('./cifar_GoogLeNet_conv1.pth'))
    conv2.load_state_dict(torch.load('./cifar_GoogLeNet_conv2.pth'))
    conv3.load_state_dict(torch.load('./cifar_GoogLeNet_conv3.pth'))
    inception3a_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception3a_branch1.pth'))
    inception3a_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception3a_branch2.pth'))
    inception3a_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception3a_branch3.pth'))
    inception3a_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception3a_branch4.pth'))
    inception3b_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception3b_branch1.pth'))
    inception3b_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception3b_branch2.pth'))
    inception3b_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception3b_branch3.pth'))
    inception3b_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception3b_branch4.pth'))
    inception4a_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception4a_branch1.pth'))
    inception4a_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception4a_branch2.pth'))
    inception4a_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception4a_branch3.pth'))
    inception4a_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception4a_branch4.pth'))
    inception4b_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception4b_branch1.pth'))
    inception4b_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception4b_branch2.pth'))
    inception4b_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception4b_branch3.pth'))
    inception4b_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception4b_branch4.pth'))
    inception4c_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception4c_branch1.pth'))
    inception4c_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception4c_branch2.pth'))
    inception4c_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception4c_branch3.pth'))
    inception4c_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception4c_branch4.pth'))
    inception4d_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception4d_branch1.pth'))
    inception4d_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception4d_branch2.pth'))
    inception4d_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception4d_branch3.pth'))
    inception4d_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception4d_branch4.pth'))
    inception4e_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception4e_branch1.pth'))
    inception4e_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception4e_branch2.pth'))
    inception4e_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception4e_branch3.pth'))
    inception4e_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception4e_branch4.pth'))
    inception5a_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception5a_branch1.pth'))
    inception5a_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception5a_branch2.pth'))
    inception5a_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception5a_branch3.pth'))
    inception5a_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception5a_branch4.pth'))
    inception5b_branch1.load_state_dict(torch.load('./cifar_GoogLeNet_inception5b_branch1.pth'))
    inception5b_branch2.load_state_dict(torch.load('./cifar_GoogLeNet_inception5b_branch2.pth'))
    inception5b_branch3.load_state_dict(torch.load('./cifar_GoogLeNet_inception5b_branch3.pth'))
    inception5b_branch4.load_state_dict(torch.load('./cifar_GoogLeNet_inception5b_branch4.pth'))
    #inception_aux1.load_state_dict(torch.load('./cifar_GoogLeNet_inception_aux1.pth'))
    #inception_aux2.load_state_dict(torch.load('./cifar_GoogLeNet_inception_aux2.pth'))
    fully_connected.load_state_dict(torch.load('./cifar_GoogLeNet_fully_connected.pth'))

    correct = 0
    total = 0
    avg_time = 0
    time_conv1 = 0
    time_conv2 = 0
    time_conv3 = 0
    time_inception3a = 0
    time_inception3b = 0
    time_inception4a = 0
    time_inception4b = 0
    time_inception4c = 0
    time_inception4d = 0
    time_inception4e = 0
    time_inception5a = 0
    time_inception5b = 0
    time_fully_connected = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data[0], data[1]
            
            total_start = time.time()
            
            # conv1 and max pool
            start = time.time()
            outputs = conv1(images)
            outputs = conv1_maxpool(outputs)
            time_conv1 += time.time() - start
            
            # conv2
            start = time.time()
            outputs = conv2(outputs)
            time_conv2 += time.time() - start
            
            # conv3 and max pool
            start = time.time()
            outputs = conv3(outputs)
            outputs = conv3_maxpool(outputs)
            time_conv3 += time.time() - start
            
            # inception3a
            start = time.time()
            branch1 = inception3a_branch1(outputs)
            branch2 = inception3a_branch2(outputs)
            branch3 = inception3a_branch3(outputs)
            branch4 = inception3a_branch4(outputs)
            outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
            time_inception3a += time.time() - start
            
            # inception3b and max pool
            start = time.time()
            branch1 = inception3b_branch1(outputs)
            branch2 = inception3b_branch2(outputs)
            branch3 = inception3b_branch3(outputs)
            branch4 = inception3b_branch4(outputs)
            outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
            outputs = inception3b_maxpool(outputs)
            time_inception3b += time.time() - start
            
            # inception4a
            start = time.time()
            branch1 = inception4a_branch1(outputs)
            branch2 = inception4a_branch2(outputs)
            branch3 = inception4a_branch3(outputs)
            branch4 = inception4a_branch4(outputs)
            outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
            time_inception4a += time.time() - start
            
            # inception4b
            start = time.time()
            branch1 = inception4b_branch1(outputs)
            branch2 = inception4b_branch2(outputs)
            branch3 = inception4b_branch3(outputs)
            branch4 = inception4b_branch4(outputs)
            outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
            time_inception4b += time.time() - start
            
            # inception4c
            start = time.time()
            branch1 = inception4c_branch1(outputs)
            branch2 = inception4c_branch2(outputs)
            branch3 = inception4c_branch3(outputs)
            branch4 = inception4c_branch4(outputs)
            outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
            time_inception4c += time.time() - start
            
            # inception4d
            start = time.time()
            branch1 = inception4d_branch1(outputs)
            branch2 = inception4d_branch2(outputs)
            branch3 = inception4d_branch3(outputs)
            branch4 = inception4d_branch4(outputs)
            outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
            time_inception4d += time.time() - start
            
            # inception4e
            start = time.time()
            branch1 = inception4e_branch1(outputs)
            branch2 = inception4e_branch2(outputs)
            branch3 = inception4e_branch3(outputs)
            branch4 = inception4e_branch4(outputs)
            outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
            outputs = inception4e_maxpool(outputs)
            time_inception4e += time.time() - start
            
            # inception5a
            start = time.time()
            branch1 = inception5a_branch1(outputs)
            branch2 = inception5a_branch2(outputs)
            branch3 = inception5a_branch3(outputs)
            branch4 = inception5a_branch4(outputs)
            outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
            time_inception5a += time.time() - start
            
            # inception5b
            start = time.time()
            branch1 = inception5b_branch1(outputs)
            branch2 = inception5b_branch2(outputs)
            branch3 = inception5b_branch3(outputs)
            branch4 = inception5b_branch4(outputs)
            outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
            time_inception5b += time.time() - start
            
            # avg pool and fully_connected
            start = time.time()
            outputs = avgpool(outputs)
            outputs = fully_connected(outputs)
            time_fully_connected += time.time() - start
            
            _, predicted = torch.max(outputs.data, 1)
            avg_time += time.time() - total_start
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i == 99:
                print('total: ', total)
                print('correct: ', correct)
                print('avg_time: ', avg_time / 100)
                print("time_conv1: ", time_conv1 / 100)
                print("time_conv2: ", time_conv2 / 100)
                print("time_conv3: ", time_conv3 / 100)
                print("time_inception3a: ", time_inception3a / 100)
                print("time_inception3b: ", time_inception3b / 100)
                print("time_inception4a: ", time_inception4a / 100)
                print("time_inception4b: ", time_inception4b / 100)
                print("time_inception4c: ", time_inception4c / 100)
                print("time_inception4d: ", time_inception4d / 100)
                print("time_inception4e: ", time_inception4e / 100)
                print("time_inception5a: ", time_inception5a / 100)
                print("time_inception5b: ", time_inception5b / 100)
                print("time_fully_connected: ", time_fully_connected / 100)
                break
            if i % 100 == 99:
                print('[%3d/%3d] tested' % (i + 1, len(testset) / 10))

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))