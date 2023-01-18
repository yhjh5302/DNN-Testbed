"""
    목표 : 서로 다른 디바이스 간 통신시 파티션에대해 올바른 순서로 array 합치기

    주어진 것 :  unique id (service id and partition id) 
                각 id, device 간 매핑 정보 (alloc_device, alloc_id)
                id의 predecessor, successor 정보 (pred, succ)
                각 파티션별 input, output slicing 정보 (sender/receiver start/end row)
                이외의 정보는 밑의 test config
                아주 간단한 형태의 CNN 구조를 가정
    
    getTag 함수를 통해 디바이스간 통신시 필요한 고유 tag를 얻는다.
     -> sender의 id와 보내는 index를 통해 결정

    run ->  - 하나의 파티션에 대한 계산을 수행하려면 해당 파티션에 대한 recv와 이전에 수행하던 send를 모두 수행해야 진행한다.
            - send는 상관없이 recv만 완료하면 수행하도록 하려 했으나 4번 레이어 이후로 수행하지 않고 끝나는 버그가 존재하여
              임의로 위와 같이 수행했다.

"""

#!/usr/bin/env python
import os, time, argparse, zmq, numpy as np, io
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

####################  test config  #####################

SIZE_PADDING = 1
SIZE_MAIN = 3
SIZE_ALL = 9

ROW_DIM = 0

                                    #    1  4
unique_id = [0,1,2,3,4,5,6,7]       # 0  2  5  7
                                    #    3  6
alloc_device = [0,0,1,2,0,1,2,0]
alloc_id = [[0,1,4,7],[2,5],[3,6]]

pred = [[],[0],[0],[0],[1,2],[1,2,3],[2,3],[4,5,6]]
succ = [[1,2,3],[4,5],[4,5,6],[5,6],[7],[7],[7],[]]

sender_start_row = [[0,2,5],[0,3],[0,1,4],[0,1],[0],[1],[1],[]]
sender_end_row = [[4,7,9],[3,4],[1,4,5],[1,4],[3],[4],[4],[]]
receiver_start_row = [[],[0],[0],[0],[0,3],[0,1,4],[0,1],[0,3,6]]
receiver_end_row = [[],[4],[5],[4],[3,4],[1,4,5],[1,4],[3,6,9]]


#########################################################

def getTag(id:int, order:int):
    return (id << 2) | order

def isStart(id):
    return len(pred[id]) == 0

def isEnd(id):
    return len(succ[id]) == 0

def doCalc(id,t):
    print("Do calc on ", id, t)

def run(rank):
    obj_lst = []
    same_dev_tensor = {}
    for id in alloc_id[rank]:
        #prepare tensors
        tensor = []
        if isStart(id):
            tensor.append(torch.rand(size=(SIZE_ALL,SIZE_ALL)))
            #for debug
            #print(tensor[0])
        else:
            for i in range(len(receiver_start_row[id])):
                tensor.append(torch.empty(size=(receiver_end_row[id][i] - receiver_start_row[id][i],SIZE_ALL)))

        #receive tensor
        for i in range(len(pred[id])):
            pr = pred[id][i]
            p2s_idx = succ[pr].index(id)
            tag = getTag(pr,p2s_idx)
            if rank == alloc_device[pr]:
                tensor[i] = same_dev_tensor[tag]
            else:
                obj = dist.irecv(tensor=tensor[i], src=alloc_device[pr], tag=tag)
                obj_lst.append(obj)
        
        #for sync
        for obj in obj_lst:
            #for debug
            #print("Waiting... ",rank,id,obj)
            obj.wait()
        obj_lst.clear()

        #partition calc
        tensor = torch.cat(tensor, dim=ROW_DIM)
        doCalc(id,tensor)

        #send tensor
        for i in range(len(succ[id])):
            su = succ[id][i]
            tag = getTag(id,i)
            if rank == alloc_device[su]:
                same_dev_tensor[tag] = tensor[sender_start_row[id][i]:sender_end_row[id][i]]
            else:
                obj = dist.isend(tensor=tensor[sender_start_row[id][i]:sender_end_row[id][i]], dst=alloc_device[su], tag=tag)
                obj_lst.append(obj)

        #for debug
        #if isEnd(id):
        #    print(tensor)

    for obj in obj_lst:
        print("Waiting... ",rank,id,obj)
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
    
