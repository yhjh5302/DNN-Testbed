#!/usr/bin/env python
import os, time, argparse, zmq, numpy as np, io
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """ Distributed function to be implemented later. """
    if rank == 0:
        # Send the tensor to process 1
        context = zmq.Context()
        sock = context.socket(zmq.PAIR)
        sock.bind("tcp://*:30000")
    elif rank == 1:
        # Receive tensor from process 0
        context = zmq.Context()
        sock = context.socket(zmq.PAIR)
        sock.connect("tcp://192.168.1.2:30000")
    if rank == 2:
        # Send the tensor to process 1
        context = zmq.Context()
        sock = context.socket(zmq.PAIR)
        sock.bind("tcp://*:30001")
    elif rank == 3:
        # Receive tensor from process 0
        context = zmq.Context()
        sock = context.socket(zmq.PAIR)
        sock.connect("tcp://192.168.1.2:30001")
    tensor = torch.randint(low=0,high=100,size=(20000,), dtype=torch.int8)
    print(tensor.dtype, tensor)
    if rank == 0:
        # Send the tensor to process 1
        start = time.time()
        for i in range(10000):
            tensor = torch.randint(low=0,high=100,size=(20000,), dtype=torch.int8)
            dist.send(tensor=tensor, dst=1)
        print("dist", time.time()-start)
        start = time.time()
        for i in range(10000):
            tensor = torch.randint(low=0,high=100,size=(20000,), dtype=torch.int8)
            packet = sock.recv()
            # print("packet", len(packet))
            header = packet[:1].decode('utf-8')
            data = np.load(io.BytesIO(packet[1:]), allow_pickle=True)
            data = torch.Tensor(data)
        print("zmq", time.time()-start)
    elif rank == 1:
        # Receive tensor from process 0
        start = time.time()
        for i in range(10000):
            tensor = torch.empty(20000)
            dist.recv(tensor=tensor, src=0)
        print("dist", time.time()-start)
        start = time.time()
        for i in range(10000):
            tensor = torch.randint(low=0,high=100,size=(20000,), dtype=torch.int8)
            f = io.BytesIO()
            np.save(f, tensor.numpy(), allow_pickle=True)
            f.seek(0)
            out = "d".encode('utf-8') + f.read()
            # print("out", len(out))
            sock.send(out)
        print("zmq", time.time()-start)
    elif rank == 2:
        # Send the tensor to process 1
        start = time.time()
        for i in range(10000):
            tensor = torch.randint(low=0,high=100,size=(20000,), dtype=torch.int8)
            dist.send(tensor=tensor, dst=3)
        print("dist", time.time()-start)
        start = time.time()
        for i in range(10000):
            tensor = torch.randint(low=0,high=100,size=(20000,), dtype=torch.int8)
            packet = sock.recv()
            # print("packet", len(packet))
            header = packet[:1].decode('utf-8')
            data = np.load(io.BytesIO(packet[1:]), allow_pickle=True)
            data = torch.Tensor(data)
        print("zmq", time.time()-start)
    elif rank == 3:
        # Receive tensor from process 0
        start = time.time()
        for i in range(10000):
            tensor = torch.empty(20000)
            dist.recv(tensor=tensor, src=2)
        print("dist", time.time()-start)
        start = time.time()
        for i in range(10000):
            tensor = torch.randint(low=0,high=100,size=(20000,), dtype=torch.int8)
            f = io.BytesIO()
            np.save(f, tensor.numpy(), allow_pickle=True)
            f.seek(0)
            out = "d".encode('utf-8') + f.read()
            # print("out", len(out))
            sock.send(out)
        print("zmq", time.time()-start)
    else:
        print(time.time()-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Torch')
    parser.add_argument('--size', default=4, type=int, help='')
    parser.add_argument('--rank', default=0, choices=[0,1,2,3], type=int, help='', required=True)
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '192.168.1.2'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='gloo', rank=args.rank, world_size=args.size)
    time.sleep(1)
    run(args.rank, args.size)
