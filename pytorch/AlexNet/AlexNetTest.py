import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import time

PATH = './'
num_classes=1000

features1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
)
features2 = nn.Sequential(
    nn.Conv2d(64, 192, kernel_size=5, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
)
features3 = nn.Sequential(
    nn.Conv2d(192, 384, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
)
features4 = nn.Sequential(
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
)
features5 = nn.Sequential(
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.AdaptiveAvgPool2d((6, 6)),
    nn.Dropout(),
)
classifier1 = nn.Sequential(
    nn.Linear(256 * 6 * 6, 4096),
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
    
    features1.load_state_dict(torch.load(PATH + 'cifar_AlexNet_features1.pth'))
    features2.load_state_dict(torch.load(PATH + 'cifar_AlexNet_features2.pth'))
    features3.load_state_dict(torch.load(PATH + 'cifar_AlexNet_features3.pth'))
    features4.load_state_dict(torch.load(PATH + 'cifar_AlexNet_features4.pth'))
    features5.load_state_dict(torch.load(PATH + 'cifar_AlexNet_features5.pth'))
    classifier1.load_state_dict(torch.load(PATH + 'cifar_AlexNet_classifier1.pth'))
    classifier2.load_state_dict(torch.load(PATH + 'cifar_AlexNet_classifier2.pth'))
    classifier3.load_state_dict(torch.load(PATH + 'cifar_AlexNet_classifier3.pth'))
    print("Pretrained model loading done!")

    correct = 0
    total = 0
    avg_time = 0
    time_features1 = 0
    time_features2 = 0
    time_features3 = 0
    time_features4 = 0
    time_features5 = 0
    time_classifier1 = 0
    time_classifier2 = 0
    time_classifier3 = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data[0], data[1]
            
            total_start = time.time()
            
            # features1
            start = time.time()
            outputs = features1(images)
            time_features1 += time.time() - start
            
            # features2
            start = time.time()
            outputs = features2(outputs)
            time_features2 += time.time() - start
            
            # features3
            start = time.time()
            outputs = features3(outputs)
            time_features3 += time.time() - start
            
            # features4
            start = time.time()
            outputs = features4(outputs)
            time_features4 += time.time() - start
            
            # features5
            start = time.time()
            outputs = features5(outputs)
            time_features5 += time.time() - start
            
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
                print("time_features1: ", time_features1 / 100)
                print("time_features2: ", time_features2 / 100)
                print("time_features3: ", time_features3 / 100)
                print("time_features4: ", time_features4 / 100)
                print("time_features5: ", time_features5 / 100)
                print("time_classifier1: ", time_classifier1 / 100)
                print("time_classifier2: ", time_classifier2 / 100)
                print("time_classifier3: ", time_classifier3 / 100)
                break
            if i % 100 == 99:
                print('[%3d/%3d] tested' % (i + 1, len(testset) / 10))

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))