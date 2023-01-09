import threading, time, argparse
import torch
import torch.distributed as dist

SCHEDULE_TAG = -1

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
        if len(schedule_list) > 0:
            with schedule_lock:
                src, tag, input_shape,  = schedule_list.pop(0)
        # input data를 받아 mapping하는 부분
        for 
            data = torch.zeros(input_shape)
            dist.recv(tensor=data, src=src, tag=tag)
        input_data = torch.cat()
        with data_lock:
            data_list.append(input_data)

def send_thread(schedule_list, schedule_lock, data_list, data_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(schedule_list) > 0 and len(data_list):
            with schedule_lock:
                src, tag, input_shape,  = schedule_list.pop(0)
            with data_lock:
                data = data_list.pop(0)
            # output data를 받아 조각내고 전송하는 부분
            src, tag, input_shape,  = schedule_list.pop(0)
            dist.send(tensor=data, dst=, tag=DATA_TAG)
        else:
            time.sleep(0.00001)

def schedule_thread(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, _stop_event):
    while _stop_event.is_set() == False:
        packet = sock.recv()
        header = packet[:1].decode('utf-8')
        data = packet[1:]
        if header == "d":
            with data_lock:
                data_list.append(np.load(io.BytesIO(data), allow_pickle=True))
        elif header == "s":
            with schedule_lock:
                schedule_list.append(np.load(io.BytesIO(data), allow_pickle=True))

def edge_scheduler(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(data_list) > 0:
            with data_lock:
                data = data_list.pop(0)
            f = io.BytesIO()
            np.save(f, data, allow_pickle=True)
            f.seek(0)
            out = "d".encode('utf-8') + f.read()
            sock.send(out)
        elif len(requests_list) > 0:
            with requests_lock:
                data = requests_list.pop(0)
            f = io.BytesIO()
            np.save(f, data, allow_pickle=True)
            f.seek(0)
            out = "r".encode('utf-8') + f.read()
            sock.send(out)
        else:
            time.sleep(0.00001)