import tensorflow as tf
import numpy as np
import socket
import io
import time
import threading

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
    period = recv_schedule(scheduler_sock)
    schduled_time = time.time()
    while _stop_event.is_set() == False:
        if len(data_list) > 0:
            with lock:
                return data_list.pop(0)
        else:
            if time.time() - schduled_time > period:
                send_done(scheduler_sock)
                schduled_time = recv_schedule(scheduler_sock)
            time.sleep(0.001)

def recv_data(conn, data_list, lock, _stop_event):
    try:
        while True:
            length = int(conn.recv(4096).decode())
            conn.send('Ack'.encode())
            data = bytearray()
            while len(data) < length:
                data.extend(conn.recv(4096))
            with lock:
                data_list.append(np.load(io.BytesIO(data), allow_pickle=True))
    except:
        _stop_event.set()

def send_data(sock, data_list, lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(data_list) > 0:
            with lock:
                data = data_list.pop(0)
            start = time.time()
            f = io.BytesIO()
            np.save(f, data, allow_pickle=True)
            f.seek(0)
            out = f.read()
            sock.send(str(len(out)).encode())
            Ack = sock.recv(4096).decode()
            if Ack == 'Ack':
                sock.sendall(out)
        else:
            time.sleep(0.001)

def send_done(sock):
    sock.send("Done".encode())

def recv_schedule(sock):
    return float(sock.recv(4096).decode())