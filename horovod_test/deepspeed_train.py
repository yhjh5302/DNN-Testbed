from models import *
import deepspeed


def add_argument():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('-b',
                        '--batch_size',
                        default=64,
                        type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('-e',
                        '--epochs',
                        default=10,
                        type=int,
                        help='number of total epochs (default: 10)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--log-interval',
                        type=int,
                        default=1,
                        help="output logging information at a given interval")

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


# Define dataset
transform = transforms.Compose([transforms.Resize(size=(224,224),interpolation=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True, download=True, transform=transform)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == '__main__':
    # Initialize Deepspeed
    args = add_argument()
    deepspeed.init_distributed()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))
    
    # Build model
    model = AlexNet().cuda()
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.state_dict(), training_data=train_dataset)

    # import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    start = time.time()
    for epoch in range(1):   # repeat process with same data
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # receive inputs from data
            inputs, labels = data[0].to(model_engine.local_rank), data[1].to(model_engine.local_rank)
            
            # gradient set to zero
            optimizer.zero_grad()

            # forward and back prop and optimize
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)
            model_engine.backward(loss)
            model_engine.step()

            # print progress
            running_loss += loss.item()
            if i % args.log_interval == args.log_interval - 1:
                print('[%2d/%2d,%4d/%4d] loss: %.3f' % (epoch + 1, args.epochs, i + 1, len(train_dataset)/args.batch_size, running_loss / args.log_interval))
                running_loss = 0.0

    torch.cuda.synchronize()
    print('Finished Training, Took {:3f} sec'.format(time.time() - start))

    #### save trained model
    # PATH = './'
    # torch.save(model.state_dict(), PATH + 'cifar_AlexNet.pth')
    # print('Model Saved')