from models import *
from typing import Dict
import ray.train.torch as RayTrainTorch
import ray.air.config as RayAirConfig
import ray.air.session as RayAirSession


# Define dataset
transform = transforms.Compose([transforms.Resize(size=(224,224),interpolation=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True, download=True, transform=transform)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train_func(config: Dict):
    device = torch.device("cuda:"+str(RayAirSession.get_local_rank()) if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    # Partition dataset among workers using DistributedSampler
    batch_size = config["batch_size"]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    train_loader = RayTrainTorch.prepare_data_loader(train_loader)
    
    # Build model
    model = AlexNet().cuda()
    model = RayTrainTorch.prepare_model(model)

    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

    epoch_size = config["epoch_size"]
    verbose = config["verbose"]
    start = time.time()
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

    torch.cuda.synchronize()
    print('Finished Training, Took {:3f} sec'.format(time.time() - start))

    #### save trained model
    # PATH = './'
    # torch.save(model.state_dict(), PATH + 'cifar_AlexNet.pth')
    # print('Model Saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--master_addr', default='localhost', type=str, help='Master node ip address')
    parser.add_argument('--master_port', default='30000', type=str, help='Master node port')
    parser.add_argument('--ifname', default='ens3f0', type=str, help='Master node port')
    parser.add_argument('--backend', default='nccl', choices=['gloo', 'mpi', 'nccl'], type=str, help='distributed backend')
    args = parser.parse_args()

    trainer = RayTrainTorch.TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"lr": 1e-3, "momentum": 0.9, "batch_size": 64, "epoch_size": 10, "verbose": 1},
        scaling_config=RayAirConfig.ScalingConfig(num_workers=args.workers, use_gpu=True),
    )
    trainer.fit()