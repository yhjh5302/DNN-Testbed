#!/usr/bin/env python
import os, time, argparse, zmq, numpy as np, io
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """ Distributed function to be implemented later. """
    tensor = torch.randint(low=0,high=100,size=(20000,), dtype=torch.float32)
    print(tensor.dtype, tensor)
    if rank == 0:
        # Send the tensor to process 1
        start = 0
        for i in range(100):
            tensor = torch.randint(low=0,high=100,size=(20000,), dtype=torch.float32)
            dist.send(tensor=tensor, dst=1)
            if start == 0:
                start = time.time()
        print("dist 0-1", time.time()-start)
        time.sleep(1)
        # Send the tensor to process 1
        start = 0
        for i in range(100):
            tensor = torch.randint(low=0,high=100,size=(20000,), dtype=torch.float32)
            dist.send(tensor=tensor, dst=2)
            if start == 0:
                start = time.time()
        print("dist 0-2", time.time()-start)
    elif rank == 1:
        # Send the tensor to process 2
        start = 0
        for i in range(100):
            tensor = torch.randint(low=0,high=100,size=(20000,), dtype=torch.float32)
            dist.send(tensor=tensor, dst=2)
            if start == 0:
                start = time.time()
        print("dist 1-2", time.time()-start)
        time.sleep(1)
        # Receive tensor from process 0
        start = 0
        for i in range(100):
            tensor = torch.empty(20000)
            dist.recv(tensor=tensor, src=0)
            if start == 0:
                start = time.time()
        print("dist 1-0", time.time()-start)
    elif rank == 2:
        # Receive tensor from process 1
        start = 0
        for i in range(100):
            tensor = torch.empty(20000)
            dist.recv(tensor=tensor, src=1)
            if start == 0:
                start = time.time()
        print("dist 2-1", time.time()-start)
        time.sleep(1)
        # Receive tensor from process 0
        start = 0
        for i in range(100):
            tensor = torch.empty(20000)
            dist.recv(tensor=tensor, src=0)
            if start == 0:
                start = time.time()
        print("dist 2-0", time.time()-start)
    else:
        print(time.time()-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Torch')
    parser.add_argument('--size', default=3, type=int, help='')
    parser.add_argument('--rank', default=0, choices=[0,1,2], type=int, help='', required=True)
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '10.96.0.201'
    os.environ['MASTER_PORT'] = '30001'
    dist.init_process_group(backend='gloo', rank=args.rank, world_size=args.size)
    time.sleep(1)
    run(args.rank, args.size)
