import socket
import multiprocessing as mp
import argparse

def get_socket(idx):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.addr, args.port + idx))
    sock.listen()
    p, addr = sock.accept()
    print(idx, 'node is ready, Connected by', p)
    return p

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU time-sharing resource allocation scheduler')
    parser.add_argument('--addr', default='', type=str, help='recving address')
    parser.add_argument('--port', default=30050, type=int, help='starting port')
    parser.add_argument('--time', default=0.1, type=float, help='second')
    parser.add_argument('--p', default=[1.0], nargs='+', type=float, help='percentage of time to use, total sum must be 1')
    args = parser.parse_args()

    # handle exception
    if sum(args.p) != 1:
        raise RuntimeError('The sum of percentage of time to use must be 1. But got {}'.format(sum(args.p)), args.p)

    sockets = []
    print('Connecting %d nodes, please waiting...'%len(args.p))
    working_queue = [idx for idx in range(len(args.p))]
    with mp.Pool(processes=len(args.p)) as pool:
        sockets = list(pool.map(get_socket, working_queue))

    while True:
        for idx, sock in enumerate(sockets):
            sock.send(str(args.time * args.p[idx]).encode())
            done = sock.recv(4096).decode()
            print(idx, 'done!')

    for sock in sockets:
        sock.close()