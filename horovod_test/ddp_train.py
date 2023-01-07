from models import *
import argparse
import os

def main_worker(gpu, args):
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.num_nodes*args.num_gpus, rank=args.rank*args.num_gpus+gpu)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    # Define dataset
    transform = transforms.Compose([transforms.Resize(size=(224,224),interpolation=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True, download=True, transform=transform)

    # Partition dataset among workers using DistributedSampler
    batch_size = 16
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=args.workers, sampler=train_sampler)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    model = AlexNet()
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
    #from torchsummary import summary
    #summary(model, (3, 224, 224))

    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epoch_size = 10
    verbose = 25
    for epoch in range(epoch_size):   # repeat process with same data
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
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
            if i % verbose == verbose - 1:
                print('[%2d/%2d,%4d/%4d] loss: %.3f' % (epoch + 1, epoch_size, i + 1, len(train_dataset)/batch_size, running_loss / verbose))
                running_loss = 0.0

    print('Finished Training')

    #### save trained model
    # PATH = './'
    # torch.save(model.state_dict(), PATH + 'cifar_AlexNet.pth')
    # print('Model Saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--num_nodes', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--num_gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--dist_url', default='tcp://localhost:30000', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='gloo', choices=['gloo', 'mpi', 'nccl'], type=str, help='distributed backend')
    args = parser.parse_args()
    torch.multiprocessing.spawn(main_worker, nprocs=args.num_nodes, args=(args,))