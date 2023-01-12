import threading, time, argparse, os
import torch
import torch.distributed as dist

REQUEST_TAG = -1
SCHEDULE_TAG = -2

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def bring_data(data_list, data_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(data_list) > 0:
            with data_lock:
                data = data_list.pop(0)
            return data
        else:
            time.sleep(0.000001) # wait for data download

def recv_thread(schedule_list, schedule_lock, data_list, data_lock, _stop_event):
    while _stop_event.is_set() == False:
        # 스케줄 리스트에서 받아야할 데이터가 있으면
        if len(schedule_list) > 0:
            with schedule_lock:
                schedules = schedule_list.pop(0)
            # 스레드를 열고 input data를 동시에 받음
            data = []
            for i, (src, input_shape, tag) in enumerate(schedules):
                data.append(torch.Empty(input_shape))
                threading.Thread(target=dist.recv, kwargs={'tensor':data[i], 'src':src, 'tag':tag}).start()
            # scheduling decision에 있는 애들이 모두 받아졌으면 merge함
            input_data = torch.cat(data)
            with data_lock:
                data_list.append(input_data)
        else:
            time.sleep(0.000001)

def send_thread(schedule_list, schedule_lock, data_list, data_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(schedule_list) > 0 and len(data_list) > 0:
            with schedule_lock:
                schedules = schedule_list.pop(0)
            # output data를 받아 schedule대로 조각내고 목적지로 전송
            with data_lock:
                output_data = data_list.pop(0)
            for (dst, slice_shape, tag) in schedules:
                data = output_data[slice_shape]
                threading.Thread(target=dist.send, kwargs={'tensor':data, 'dst':dst, 'tag':tag}).start()
        else:
            time.sleep(0.000001)

def schedule_recv_thread(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, _stop_event):
    while _stop_event.is_set() == False:
        schedule_list = torch.Empty(schedule_shape)
        dist.recv(tensor=schedule_list, src=0, tag=SCHEDULE_TAG)
        with recv_schedule_lock:
            recv_schedule_list.extend(schedule_list[0])
        with send_schedule_lock:
            send_schedule_list.extend(schedule_list[1])

def edge_scheduler(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, _stop_event):
    while _stop_event.is_set() == False:
        request_list = torch.Empty(request_shape)
        dist.recv(tensor=request_list, src=None, tag=REQUEST_TAG)
        scheduling_decision = scheduling_algorithm(request) # TODO
        for schedule in scheduling_decision:
            # 만약 로컬에서 처리해야하면 로컬 schedule_list에 채워넣음.
            # 아니면 해당 device로 보냄.
            pass

def processing(inputs):
    return 0