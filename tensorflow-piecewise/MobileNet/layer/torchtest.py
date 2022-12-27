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
        sock = context.socket(zmq.REP)
        sock.bind("tcp://*:30000")
    elif rank == 1:
        # Receive tensor from process 0
        context = zmq.Context()
        sock = context.socket(zmq.REQ)
        sock.connect("tcp://localhost:30000")
    start = time.time()
    tensor = torch.zeros(1)
    if rank == 0:
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
        # for i in range(2):
        #     packet = sock.recv()
        #     header = packet[:1].decode('utf-8')
        #     data = packet[1:]
        print(time.time()-start)
    elif rank == 1:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
        # for i in range(2):
        #     time.sleep(1)
        #     f = io.BytesIO()
        #     np.save(f, tensor.numpy(), allow_pickle=True)
        #     f.seek(0)
        #     out = "d".encode('utf-8') + f.read()
        #     sock.send(out)
        print(time.time()-start)
    else:
        print(time.time()-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Torch')
    parser.add_argument('--size', default=3, type=int, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo', rank=args.rank, world_size=args.size)
    time.sleep(1)
    run(args.rank, args.size)
