import threading, time, argparse
import torch
import torch.distributed as dist

SCHEDULE = -1

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
                src, dst, layer = schedule_list.pop(0)
        src, dst, input_shape = schedule_list.pop()
        data = torch.zeros(input_shape)
        src = dist.recv(tensor=data, tag=0)
        with data_lock:
            data_list.append((src, data))

def send_thread(schedule_list, schedule_lock, data_list, data_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(data_list) > 0:
            with data_lock:
                data = data_list.pop(0)
            dist.send(tensor=data, dst=, tag=0)
        elif len(schedule_list) > 0:
            with schedule_lock:
                data = schedule_list.pop(0)
            f = io.BytesIO()
            np.save(f, data, allow_pickle=True)
            f.seek(0)
            out = "s".encode('utf-8') + f.read()
            sock.send(out)
        else:
            time.sleep(0.00001)

def device_recv_data(sock, data_list, data_lock, schedule_list, schedule_lock, _stop_event):
    try:
        while True:
            packet = sock.recv()
            header = packet[:1].decode('utf-8')
            data = packet[1:]
            if header == "d":
                with data_lock:
                    data_list.append(np.load(io.BytesIO(data), allow_pickle=True))
            elif header == "s":
                with schedule_lock:
                    schedule_list.append(np.load(io.BytesIO(data), allow_pickle=True))
    except:
        _stop_event.set()
        raise RuntimeError('ERROR: something wrong with previous node', sock, 'in recv_data!')

def device_send_data(sock, data_list, data_lock, requests_list, requests_lock, _stop_event):
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

def recv_schedule(sock):
    return (sock.recv(4096).decode() == 'process')