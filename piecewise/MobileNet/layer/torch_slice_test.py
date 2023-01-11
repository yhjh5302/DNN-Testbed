#!/usr/bin/env python
import os, time, argparse, zmq, numpy as np, io
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """ Distributed function to be implemented later. """
    start_t = time.time()
    tensor = torch.randint(low=0,high=100,size=(1,1000,27,27), dtype=torch.float16)
    print(tensor.dtype, tensor.shape)
    if rank == 0:
        # Send the tensor to process 1
        tensor = torch.randint(low=0,high=100,size=(1,1000,27,27), dtype=torch.float16)
        dist.send(tensor=tensor, dst=1)
        # Send the tensor to process 2
        tensor = torch.randint(low=0,high=100,size=(1,1000,27,27), dtype=torch.float16)
        dist.send(tensor=tensor, dst=2)
        # Receive tensor from process 1
        tensor = torch.empty(size=(1,1000,27,27), dtype=torch.float16)
        dist.recv(tensor=tensor, src=1)
        # Receive tensor from process 2
        tensor = torch.empty(size=(1,1000,27,27), dtype=torch.float16)
        dist.recv(tensor=tensor, src=2)
    elif rank == 1:
        # Receive tensor from process 0
        tensor = torch.empty(size=(1,1000,27,27), dtype=torch.float16)
        dist.recv(tensor=tensor, src=0)
        # Send the tensor to process 0
        tensor = torch.randint(low=0,high=100,size=(1,1000,27,27), dtype=torch.float16)
        dist.send(tensor=tensor, dst=0)
    elif rank == 2:
        # Receive tensor from process 0
        tensor = torch.empty(size=(1,1000,27,27), dtype=torch.float16)
        dist.recv(tensor=tensor, src=0)
        # Send the tensor to process 0
        tensor = torch.randint(low=0,high=100,size=(1,1000,27,27), dtype=torch.float16)
        dist.send(tensor=tensor, dst=0)
    else:
        raise RuntimeError("rank must be 0~2!!")
    print("Took {:3f} sec".format(time.time() - start_t))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Torch')
    parser.add_argument('--size', default=3, type=int, help='')
    parser.add_argument('--rank', default=0, choices=[0,1,2], type=int, help='', required=True)
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '30000'
    dist.init_process_group(backend='gloo', rank=args.rank, world_size=args.size)
    run(args.rank, args.size)
