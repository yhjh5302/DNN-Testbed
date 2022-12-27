import numpy as np
import zmq
import io
import time
import threading
import struct
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def connection_setup(sock_list, _stop_event):
    while _stop_event.is_set() == False:
        # server open and connect
        context = zmq.Context()
        sock = context.socket(zmq.REP)
        sock.bind("tcp://{}:{}".format("*", 30000))
        sock_list.append(sock)

def bring_data(data_list, data_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(data_list) > 0:
            with data_lock:
                data = data_list.pop(0)
            return data
        else:
            time.sleep(0.00001) # wait for data download

def edge_recv_data(sock, data_list, data_lock, requests_list, requests_lock, _stop_event):
    try:
        while True:
            packet = sock.recv()
            header = packet[:1].decode('utf-8')
            data = packet[1:]
            if header == "d":
                with data_lock:
                    data_list.append(np.load(io.BytesIO(data), allow_pickle=True))
            elif header == "r":
                with requests_lock:
                    requests_list.append(np.load(io.BytesIO(data), allow_pickle=True))
    except:
        _stop_event.set()
        raise RuntimeError('ERROR: something wrong with previous node', sock, 'in recv_data!')

def edge_send_data(sock, data_list, data_lock, schedule_list, schedule_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(data_list) > 0:
            with data_lock:
                data = data_list.pop(0)
            f = io.BytesIO()
            np.save(f, data, allow_pickle=True)
            f.seek(0)
            out = "d".encode('utf-8') + f.read()
            sock.send(out)
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