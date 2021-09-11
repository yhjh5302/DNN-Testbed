import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import time

PATH = './'
num_classes=1000

features1_conv1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
)
features1_conv2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
)
features2_conv1 = nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
)
features2_conv2 = nn.Sequential(
    nn.Conv2d(128, 128, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
)
features3_conv1 = nn.Sequential(
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
)
features3_conv2 = nn.Sequential(
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
)
features3_conv3 = nn.Sequential(
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
)
features4_conv1 = nn.Sequential(
    nn.Conv2d(256, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
)
features4_conv2 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
)
features4_conv3 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
)
features5_conv1 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
)
features5_conv2 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
)
features5_conv3 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.AdaptiveAvgPool2d((7, 7)),
    nn.Dropout(),
)
classifier1 = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
)
classifier2 = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
)
classifier3 = nn.Sequential(
    nn.Linear(4096, 1000),
)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize(size=(224,224),interpolation=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    features1_conv1.load_state_dict(torch.load(PATH + 'cifar_VGGNet_features1_conv1.pth'))
    features1_conv2.load_state_dict(torch.load(PATH + 'cifar_VGGNet_features1_conv2.pth'))
    features2_conv1.load_state_dict(torch.load(PATH + 'cifar_VGGNet_features2_conv1.pth'))
    features2_conv2.load_state_dict(torch.load(PATH + 'cifar_VGGNet_features2_conv2.pth'))
    features3_conv1.load_state_dict(torch.load(PATH + 'cifar_VGGNet_features3_conv1.pth'))
    features3_conv2.load_state_dict(torch.load(PATH + 'cifar_VGGNet_features3_conv2.pth'))
    features3_conv3.load_state_dict(torch.load(PATH + 'cifar_VGGNet_features3_conv3.pth'))
    features4_conv1.load_state_dict(torch.load(PATH + 'cifar_VGGNet_features4_conv1.pth'))
    features4_conv2.load_state_dict(torch.load(PATH + 'cifar_VGGNet_features4_conv2.pth'))
    features4_conv3.load_state_dict(torch.load(PATH + 'cifar_VGGNet_features4_conv3.pth'))
    features5_conv1.load_state_dict(torch.load(PATH + 'cifar_VGGNet_features5_conv1.pth'))
    features5_conv2.load_state_dict(torch.load(PATH + 'cifar_VGGNet_features5_conv2.pth'))
    features5_conv3.load_state_dict(torch.load(PATH + 'cifar_VGGNet_features5_conv3.pth'))
    classifier1.load_state_dict(torch.load(PATH + 'cifar_VGGNet_classifier1.pth'))
    classifier2.load_state_dict(torch.load(PATH + 'cifar_VGGNet_classifier2.pth'))
    classifier3.load_state_dict(torch.load(PATH + 'cifar_VGGNet_classifier3.pth'))
    print("Pretrained model loading done!")

    correct = 0
    total = 0
    avg_time = 0
    time_features1_conv1 = 0
    time_features1_conv2 = 0
    time_features2_conv1 = 0
    time_features2_conv2 = 0
    time_features3_conv1 = 0
    time_features3_conv2 = 0
    time_features3_conv3 = 0
    time_features4_conv1 = 0
    time_features4_conv2 = 0
    time_features4_conv3 = 0
    time_features5_conv1 = 0
    time_features5_conv2 = 0
    time_features5_conv3 = 0
    time_classifier1 = 0
    time_classifier2 = 0
    time_classifier3 = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data[0], data[1]
            
            total_start = time.time()
            
            # features1_conv1
            start = time.time()
            outputs = features1_conv1(images)
            time_features1_conv1 += time.time() - start
            
            # features1_conv2
            start = time.time()
            outputs = features1_conv2(outputs)
            time_features1_conv2 += time.time() - start
            
            # features2_conv1
            start = time.time()
            outputs = features2_conv1(outputs)
            time_features2_conv1 += time.time() - start
            
            # features2_conv2
            start = time.time()
            outputs = features2_conv2(outputs)
            time_features2_conv2 += time.time() - start
            
            # features3_conv1
            start = time.time()
            outputs = features3_conv1(outputs)
            time_features3_conv1 += time.time() - start
            
            # features3_conv2
            start = time.time()
            outputs = features3_conv2(outputs)
            time_features3_conv2 += time.time() - start
            
            # features3_conv3
            start = time.time()
            outputs = features3_conv3(outputs)
            time_features3_conv3 += time.time() - start
            
            # features4_conv1
            start = time.time()
            outputs = features4_conv1(outputs)
            time_features4_conv1 += time.time() - start
            
            # features4_conv2
            start = time.time()
            outputs = features4_conv2(outputs)
            time_features4_conv2 += time.time() - start
            
            # features4_conv3
            start = time.time()
            outputs = features4_conv3(outputs)
            time_features4_conv3 += time.time() - start
            
            # features5_conv1
            start = time.time()
            outputs = features5_conv1(outputs)
            time_features5_conv1 += time.time() - start
            
            # features5_conv2
            start = time.time()
            outputs = features5_conv2(outputs)
            time_features5_conv2 += time.time() - start
            
            # features5_conv3
            start = time.time()
            outputs = features5_conv3(outputs)
            time_features5_conv3 += time.time() - start
            
            outputs = torch.flatten(outputs, 1)
            
            # classifier1
            start = time.time()
            outputs = classifier1(outputs)
            time_classifier1 += time.time() - start
            
            # classifier2
            start = time.time()
            outputs = classifier2(outputs)
            time_classifier2 += time.time() - start
            
            # classifier3
            start = time.time()
            outputs = classifier3(outputs)
            time_classifier3 += time.time() - start
            
            _, predicted = torch.max(outputs.data, 1)
            avg_time += time.time() - total_start
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i == 99:
                print('total: ', total)
                print('correct: ', correct)
                print('avg_time: ', avg_time / 100)
                print("time_features1_conv1: ", time_features1_conv1 / 100)
                print("time_features1_conv2: ", time_features1_conv2 / 100)
                print("time_features2_conv1: ", time_features2_conv1 / 100)
                print("time_features2_conv2: ", time_features2_conv2 / 100)
                print("time_features3_conv1: ", time_features3_conv1 / 100)
                print("time_features3_conv2: ", time_features3_conv2 / 100)
                print("time_features3_conv3: ", time_features3_conv3 / 100)
                print("time_features4_conv1: ", time_features4_conv1 / 100)
                print("time_features4_conv2: ", time_features4_conv2 / 100)
                print("time_features4_conv3: ", time_features4_conv3 / 100)
                print("time_features5_conv1: ", time_features5_conv1 / 100)
                print("time_features5_conv2: ", time_features5_conv2 / 100)
                print("time_features5_conv3: ", time_features5_conv3 / 100)
                print("time_classifier1: ", time_classifier1 / 100)
                print("time_classifier2: ", time_classifier2 / 100)
                print("time_classifier3: ", time_classifier3 / 100)
                break
            if i % 100 == 99:
                print('[%3d/%3d] tested' % (i + 1, len(testset) / 10))

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))