import tensorflow as tf
import numpy as np
import socket
import io
import time
import threading
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

def bring_data(data_list, lock, _stop_event, scheduler_sock):
    if scheduler_sock:
        while _stop_event.is_set() == False:
            if recv_schedule(scheduler_sock): # wait for scheduler
                if len(data_list) > 0:
                    with lock:
                        return data_list.pop(0)
                else:
                    time.sleep(0.001) # wait for data download
                    send_done(scheduler_sock)
            else:
                _stop_event.set()
                raise RuntimeError('ERROR: something wrong with scheduler', scheduler_sock, 'in bring_data!')
    else:
        while _stop_event.is_set() == False:
            if len(data_list) > 0:
                with lock:
                    return data_list.pop(0)
            else:
                time.sleep(0.001) # wait for data download

def recv_data(conn, data_list, time_list, data_lock, time_lock, _stop_event):
    try:
        while True:
            length = int(conn.recv(4096).decode())
            conn.send('Ack'.encode())
            data = bytearray()
            while len(data) < length:
                data.extend(conn.recv(4096))
            conn.send('Done'.encode())
            with data_lock:
                data_list.append(np.load(io.BytesIO(data), allow_pickle=True))
            with time_lock:
                time_list.append(time.time())
    except:
        _stop_event.set()
        raise RuntimeError('ERROR: something wrong with previous node', conn, 'in recv_data!')

def send_data(sock, data_list, lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(data_list) > 0:
            with lock:
                data = data_list.pop(0)
            f = io.BytesIO()
            np.save(f, data, allow_pickle=True)
            f.seek(0)
            out = f.read()
            sock.send(str(len(out)).encode())
            Ack = sock.recv(4096).decode()
            if Ack == 'Ack':
                sock.sendall(out)
            Ack = sock.recv(4096).decode()
            if Ack != 'Done':
                print('No Done')
                _stop_event.set()
        else:
            time.sleep(0.001)

def send_done(sock):
    sock.send("Done".encode())

def recv_schedule(sock):
    return (sock.recv(4096).decode() == 'process')

def send_input(sock, data, _stop_event):
    f = io.BytesIO()
    np.save(f, data, allow_pickle=True)
    f.seek(0)
    out = f.read()
    sock.send(str(len(out)).encode())
    Ack = sock.recv(4096).decode()
    if Ack == 'Ack':
        sock.sendall(out)
    Ack = sock.recv(4096).decode()
    if Ack != 'Done':
        print('No Done')
        _stop_event.set()

def recv_output(conn, _stop_event):
    length = int(conn.recv(4096).decode())
    conn.send('Ack'.encode())
    data = bytearray()
    while len(data) < length:
        data.extend(conn.recv(4096))
    conn.send('Done'.encode())
    return np.load(io.BytesIO(data), allow_pickle=True)