#!/usr/bin/env python
import os, time, argparse, zmq, numpy as np, io
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

####################  test config  #####################
                                    #    1  4
partitoin_num = [0,1,2,3,4,5,6,7]   # 0  2  5  7
                                    #    3  6
alloc_device = [0,0,1,2,0,1,2,0]
alloc_partition = [[0,1,4,7],[2,5],[3,6]]

pred = [[],[0],[0],[0],[1,2],[1,2,3],[2,3],[4,5,6]]
succ = [[1,2,3],[4,5],[4,5,6],[5,6],[7],[7],[7],[]]

# 모두 지정
pos_next = [[0,1,2],[1,2],[0,1,2],[0,1],[1],[1],[1],[]]
pos_prev = [[],[0],[1],[2],[1,0],[2,1,0],[2,1],[1,1,1]]
#########################################################

def pack(p_num:int, order:int):
    p_num = p_num << 2
    p_num += order
    return p_num

def isStart(p_num):
    return len(pred[p_num]) == 0

def isEnd(p_num):
    return len(succ[p_num]) == 0

def doCalc(p_num,t):
    print("Do calc on ", p_num)

def run(rank):
    obj_lst = []
    for p_num in alloc_partition[rank]:
        tensor = [torch.empty(size=(3,3)) for _ in range(len(pred[p_num]))]
        for idx in range(len(pred[p_num])):
            if rank == alloc_device[pred[p_num][idx]]:
                continue
            pr = pred[p_num][idx]
            obj = dist.irecv(tensor=tensor[idx], src=alloc_device[pr], tag=pack(p_num,pos_prev[p_num][idx]))
            print("recv : ",rank,p_num,alloc_device[pr], pr, obj)
            obj_lst.append(obj)
        
        for obj in obj_lst:
            print("Waiting... ",rank,p_num,obj)
            obj.wait()
        obj_lst.clear()
        
        doCalc(p_num,tensor)

        st = torch.empty(size=(3,3))
        for idx in range(len(succ[p_num])):
            if rank == alloc_device[succ[p_num][idx]]:
                continue
            su = succ[p_num][idx]
            obj = dist.isend(tensor=st, dst=alloc_device[su], tag=pack(su,pos_next[p_num][idx]))
            print("send : ",rank,p_num,alloc_device[su],su,obj)
            obj_lst.append(obj)

        if isEnd(p_num):
            print(tensor)

    for obj in obj_lst:
        print("Waiting... ",rank,p_num,obj)
        obj.wait()
    obj_lst.clear()

def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Torch')
    parser.add_argument('--size', default=3, type=int, help='')
    #parser.add_argument('--rank', default=0, choices=[0,1,2], type=int, help='', required=True)
    args = parser.parse_args()
    size = args.size
    processes = []

    start_t = time.time()
    
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Took {:3f} sec".format(time.time() - start_t))
