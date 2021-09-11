from common import *
from AlexNetModel import *

set_gpu = True     # If you want to use GPU, set "True"
prev_addr = ''
prev_port = 30003
next_addr = 'localhost'
next_port = 30000

def processing(inputs, model):
    outputs = model(inputs)
    return outputs

if __name__ == "__main__":
    if torch.cuda.is_available() and set_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    model = nn.Sequential(
        AlexNet_classifier2(device),
        AlexNet_classifier3(device),
    ).to(device)
    print('Pretrained model loading done!')
    
    prev_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    prev_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    prev_sock.bind((prev_addr, prev_port))
    prev_sock.listen()
    p, addr = prev_sock.accept()
    print('Previous node is ready, Connected by', addr)
    
    next_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    next_sock.settimeout(600) # 10 minutes
    next_sock.connect((next_addr, next_port))
    print('Next node is ready, Connected by', next_addr)
    
    # for time record
    total = 0
    total2 = 0
    
    # for data multi-processing
    data_list = []
    lock = threading.Lock()
    _stop_event = threading.Event()
    
    proc = threading.Thread(target=recv_data, args=(p, data_list, lock, _stop_event))
    proc.start()
    
    # for debugging
    #with torch.no_grad():
    try:    # I know it isn't good for debug, but I don't want to see the socket error message...
        with torch.no_grad():
            while True:
                inputs = bring_data(data_list, lock, _stop_event)
                if torch.cuda.is_available() and set_gpu:
                    inputs = inputs.cuda()
                start = time.time()
                outputs = processing(inputs, model)
                total += time.time() - start
                send_data(next_sock, outputs)
                total2 += time.time() - start
    # for debugging
    #with torch.no_grad():
    except:    # I know it isn't good for debug, but I don't want to see the socket error message...
        print("total:", total)
        print("total2:", total2)
    proc.join()
    prev_sock.close()
    next_sock.close()