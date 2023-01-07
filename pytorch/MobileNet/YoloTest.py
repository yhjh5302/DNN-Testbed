import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import math, time

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_per_process_memory_fraction(fraction=0.2, device=device)
    torch.backends.cudnn.benchmark = True
    print(device)
    print(torch.cuda.get_device_name(0))
    half = False

    transform = transforms.Compose([transforms.Resize(size=(224,224),interpolation=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    if half == True:
        model = torch.hub.load('ultralystics/yolov5', 'yolov5s', pretrained=True).cuda().eval().half()
        test_input = torch.zeros(1,3,224,224).cuda().half()
    else:
        model = torch.hub.load('ultralystics/yolov5', 'yolov5s', pretrained=True).cuda().eval()
        test_input = torch.zeros(1,3,224,224).cuda()
    model(test_input)

    correct = 0
    total = 0
    avg_time = 0
    torch.cuda.synchronize()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data[0], data[1]
            if half == True:
                images = images.to(torch.float16).cuda()
            else:
                images = images.cuda()
            
            start = time.time()
            outputs = model(images)
            torch.cuda.synchronize()
            
            _, predicted = torch.max(outputs.data, 1)
            took = time.time() - start
            avg_time += took
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()
            print("{}th took {:3f}".format(i, took))
            if i == 99:
                print('total: ', total)
                print('correct: ', correct)
                print('avg_time: ', avg_time / 100)
                break
            # if i % 100 == 99:
            #     print('[%3d/%3d] tested' % (i + 1, len(testset) / 10))

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))