import torch
import torch.nn as nn
import socket
import io
import time
import threading

def bring_data(data_list, lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(data_list) > 0:
            with lock:
                return data_list.pop(0)
        else:
            time.sleep(0.001)

def recv_result(conn):
    length = int(conn.recv(4096).decode())
    conn.send('Request Ack'.encode())
    data = b''
    while True:
        recv = conn.recv(4096)
        data += recv
        if len(data) >= length: break
    conn.send('Done Ack'.encode())
    return torch.load(io.BytesIO(data), map_location=torch.device('cpu'))

def recv_data(conn, data_list, lock, _stop_event):
    try:
        while True:
            length = int(conn.recv(4096).decode())
            conn.send('Request Ack'.encode())
            data = b''
            while True:
                recv = conn.recv(4096)
                data += recv
                if len(data) >= length: break
            conn.send('Done Ack'.encode())
            with lock:
                data_list.append(torch.load(io.BytesIO(data), map_location=torch.device('cpu')))
    except:
        _stop_event.set()

def send_data(sock, data):
    f = io.BytesIO()
    torch.save(data, f)
    f.seek(0)
    out = f.read()
    sock.send(str(len(out)).encode())
    Ack = sock.recv(4096).decode()
    if Ack == 'Request Ack':
        sock.sendall(out)
        Ack = sock.recv(4096).decode()
        if Ack == 'Done Ack':
            return
        else:
            print('No Done Ack')
    else:
        print('No Request Ack')