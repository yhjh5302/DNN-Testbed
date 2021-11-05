from common import *
from GoogLeNetModel import *

def processing(inputs, model):
    outputs = model(inputs)
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow')
    parser.add_argument('--set_gpu', default=True, type=str2bool, help='If you want to use GPU, set "True"')
    parser.add_argument('--prev_addr', default='10.96.0.213', type=str, help='Previous node address')
    parser.add_argument('--prev_port', default=30013, type=int, help='Previous node port')
    parser.add_argument('--next_addr', default='10.96.0.200', type=str, help='Next node address')
    parser.add_argument('--next_port', default=30010, type=int, help='Next node port')
    parser.add_argument('--scheduler_addr', default='10.96.0.250', type=str, help='Scheduler address')
    parser.add_argument('--scheduler_port', default=30050, type=int, help='Scheduler port')
    parser.add_argument('--vram_limit', default=300, type=int, help='Vram limitation')
    parser.add_argument('--debug', default=100, type=int, help='How often to print debug statements')
    args = parser.parse_args()

    if args.set_gpu:
        gpu_devices = tf.config.list_physical_devices(device_type='GPU')
        if not gpu_devices:
            raise ValueError('Cannot detect physical GPU device in TF')
        tf.config.set_logical_device_configuration(gpu_devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=args.vram_limit)])
        tf.config.list_logical_devices()
    else:
        tf.config.set_visible_devices([], 'GPU')

    # model loading
    model = GoogLeNet_layer_3(name='layer3')
    model.inception5a_branch1.load_weights('./GoogLeNet_inception5a_branch1_weights')
    model.inception5a_branch2.load_weights('./GoogLeNet_inception5a_branch2_weights')
    model.inception5a_branch3.load_weights('./GoogLeNet_inception5a_branch3_weights')
    model.inception5a_branch4.load_weights('./GoogLeNet_inception5a_branch4_weights')
    model.inception5b_branch1.load_weights('./GoogLeNet_inception5b_branch1_weights')
    model.inception5b_branch2.load_weights('./GoogLeNet_inception5b_branch2_weights')
    model.inception5b_branch3.load_weights('./GoogLeNet_inception5b_branch3_weights')
    model.inception5b_branch4.load_weights('./GoogLeNet_inception5b_branch4_weights')
    model.fully_connected.load_weights('./GoogLeNet_fully_connected_weights')

    # for cuDNN loading
    model(np.zeros((1,6,6,832)))

    print('Pretrained model loading done!')

    prev_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    prev_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    prev_sock.bind((args.prev_addr, args.prev_port))
    prev_sock.listen()
    p, addr = prev_sock.accept()
    print('Previous node is ready, Connected by', addr)

    next_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    next_sock.settimeout(1000) # 1000 seconds
    next_sock.connect((args.next_addr, args.next_port))
    print('Next node is ready, Connected by', args.next_addr)

    scheduler_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    scheduler_sock.settimeout(1000) # 1000 seconds
    scheduler_sock.connect((args.scheduler_addr, args.scheduler_port))
    print('Scheduler is ready, Connected by', args.scheduler_addr)

    # for data multi-processing
    recv_data_list = []
    recv_lock = threading.Lock()
    send_data_list = []
    send_lock = threading.Lock()
    recv_time_list = []
    recv_time_lock = threading.Lock()
    _stop_event = threading.Event()
    threading.Thread(target=recv_data, args=(p, recv_data_list, recv_time_list, recv_lock, recv_time_lock, _stop_event)).start()
    threading.Thread(target=send_data, args=(next_sock, send_data_list, send_lock, _stop_event)).start()

    while True:
        inputs = bring_data(recv_data_list, recv_lock, _stop_event, scheduler_sock)
        outputs = processing(inputs, model)
        send_done(scheduler_sock)
        with send_lock:
            send_data_list.append(outputs)
        with recv_time_lock:
            print("processing time", time.time() - recv_time_list.pop(0))

    prev_sock.close()
    next_sock.close()
    scheduler_sock.close()