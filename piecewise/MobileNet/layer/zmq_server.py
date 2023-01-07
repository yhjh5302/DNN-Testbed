import zmq

context = zmq.Context()

sock = context.socket(zmq.REP)
sock.bind("tcp://*:30000")
data = sock.recv().decode()
print(len(data))