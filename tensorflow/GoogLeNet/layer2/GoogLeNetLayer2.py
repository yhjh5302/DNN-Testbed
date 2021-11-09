from common import *
from GoogLeNetModel import *

def processing(inputs, model):
    outputs = model(inputs)
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow')
    parser.add_argument('--set_gpu', default=212, type=str2bool, help='If you want to use GPU, set "True"')
    parser.add_argument('--prev_addr', default='10.96.0.212', type=str, help='Previous node address')
    parser.add_argument('--prev_port', default=30012, type=int, help='Previous node port')
    parser.add_argument('--next_addr', default='10.96.0.213', type=str, help='Next node address')
    parser.add_argument('--next_port', default=30013, type=int, help='Next node port')
    parser.add_argument('--scheduler_addr', default='10.96.0.250', type=str, help='Scheduler address')
    parser.add_argument('--scheduler_port', default=30050, type=int, help='Scheduler port')
    parser.add_argument('--vram_limit', default=100, type=int, help='Vram limitation')
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
    model = GoogLeNet_layer_2(name='layer2')
    model.inception4a_branch1.load_weights('./GoogLeNet_inception4a_branch1_weights')
    model.inception4a_branch2.load_weights('./GoogLeNet_inception4a_branch2_weights')
    model.inception4a_branch3.load_weights('./GoogLeNet_inception4a_branch3_weights')
    model.inception4a_branch4.load_weights('./GoogLeNet_inception4a_branch4_weights')
    model.inception4b_branch1.load_weights('./GoogLeNet_inception4b_branch1_weights')
    model.inception4b_branch2.load_weights('./GoogLeNet_inception4b_branch2_weights')
    model.inception4b_branch3.load_weights('./GoogLeNet_inception4b_branch3_weights')
    model.inception4b_branch4.load_weights('./GoogLeNet_inception4b_branch4_weights')
    model.inception4c_branch1.load_weights('./GoogLeNet_inception4c_branch1_weights')
    model.inception4c_branch2.load_weights('./GoogLeNet_inception4c_branch2_weights')
    model.inception4c_branch3.load_weights('./GoogLeNet_inception4c_branch3_weights')
    model.inception4c_branch4.load_weights('./GoogLeNet_inception4c_branch4_weights')
    model.inception4d_branch1.load_weights('./GoogLeNet_inception4d_branch1_weights')
    model.inception4d_branch2.load_weights('./GoogLeNet_inception4d_branch2_weights')
    model.inception4d_branch3.load_weights('./GoogLeNet_inception4d_branch3_weights')
    model.inception4d_branch4.load_weights('./GoogLeNet_inception4d_branch4_weights')
    model.inception4e_branch1.load_weights('./GoogLeNet_inception4e_branch1_weights')
    model.inception4e_branch2.load_weights('./GoogLeNet_inception4e_branch2_weights')
    model.inception4e_branch3.load_weights('./GoogLeNet_inception4e_branch3_weights')
    model.inception4e_branch4.load_weights('./GoogLeNet_inception4e_branch4_weights')

    # for cuDNN loading
    model(np.zeros((1,13,13,480)))

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

    if args.set_gpu:
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
        if args.set_gpu:
            inputs = bring_data(recv_data_list, recv_lock, _stop_event, scheduler_sock)
            outputs = processing(inputs, model)
            send_done(scheduler_sock)
        else:
            inputs = bring_data(recv_data_list, recv_lock, _stop_event)
            outputs = processing(inputs, model)
        with send_lock:
            send_data_list.append(outputs)
        with recv_time_lock:
            print("processing time", time.time() - recv_time_list.pop(0))

    prev_sock.close()
    next_sock.close()
    if args.set_gpu:
        scheduler_sock.close()