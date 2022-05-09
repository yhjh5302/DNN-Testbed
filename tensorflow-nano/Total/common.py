import tensorflow as tf
import numpy as np
import pickle
import socket
import io
import time
import threading
import argparse
import random
from time import sleep

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def bring_data(data_dict, lock, _stop_event, prob=None, init_prob=None):
    loop_count = 0
    loop_max = 100
    while _stop_event.is_set() == False:
        if data_dict['waiting_num'] > 0:
            with lock:
                if loop_count < loop_max:
                    target = random.choices(population=data_dict['partitions'], weights=prob)[0]
                else:
                    target = random.choices(population=data_dict['partitions'], weights=init_prob)[0]  # break inf loop
                    
                if data_dict['proc'][target] is not None:
                    result = data_dict['proc'][target]

                elif len(data_dict[target]) > 0:
                    result = data_dict[target].pop(0)                    
                    result =  (result, time.time())

                else:
                    result = None

                if result is not None:
                    if len(data_dict[target]) > 0:  # update new processing
                        new_data = data_dict[target].pop(0)
                        data_dict['proc'][target] = (new_data, time.time())
                    else:
                        data_dict['proc'][target] = None
                    return result
                loop_count += 1
        else:
            time.sleep(0.001) # wait for data download

def recv_data(conn, recv_data_dict, recv_data_lock, _stop_event, dag_man):
    try:
        while True:            
            length = int(conn.recv(4096).decode())
            start = time.time()
            conn.send('Ack'.encode())
            data = bytearray()
            while len(data) < length:
                data.extend(conn.recv(4096))
            conn.send('Done'.encode())
            inputs = pickle.load(io.BytesIO(data))
            cur_time = time.time()
            result = dag_man.recv_data(inputs, start, cur_time)

            if result is not None:
                with recv_data_lock:
                    target_partition = result[0][2]
                    if recv_data_dict['proc'][target_partition] is None:
                        recv_data_dict['proc'][target_partition] = (result, cur_time)
                    else:
                        recv_data_dict[target_partition].append(result)
                    recv_data_dict['waiting_num'] += 1
            
    except:
        _stop_event.set()
        raise RuntimeError('ERROR: something wrong with previous node', conn, 'in recv_data!')

def send_data(sock, data_list, lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(data_list) > 0:
            with lock:
                data = data_list.pop(0)
            f = io.BytesIO()
            # np.save(f, data, allow_pickle=True)
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
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
    # np.save(f, data, allow_pickle=True)
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
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
    # return np.load(io.BytesIO(data), allow_pickle=True)
    return pickle.load(io.BytesIO(data))


def open_resv_sock(resv_ip, resv_port):
    resv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    resv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    resv_sock.bind((resv_ip, resv_port))
    resv_sock.listen()
    conn, addr = resv_sock.accept()
    print('receive socket is ready, Connected by', addr)
    return conn, addr
    

def open_send_sock(send_addr, send_port):
    while True:
        try:
            send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            send_sock.settimeout(1000) # 1000 seconds
            send_sock.connect((send_addr, send_port))
            print('send socket is ready, Connected by', send_addr)
            break
        except ConnectionError:
            print("server is not opened try later")
            sleep(10)
    
    return send_sock, send_addr


def server_socket(resv_opt, send_opt):
    resv_conn, resv_addr = open_resv_sock(*resv_opt)
    send_sock, send_addr = open_send_sock(*send_opt)
    return resv_conn, resv_addr, send_sock, send_addr
    

def client_socket(resv_opt, send_opt):
    send_sock, send_addr = open_send_sock(*send_opt)
    resv_conn, resv_addr = open_resv_sock(*resv_opt)
    return resv_conn, resv_addr, send_sock, send_addr