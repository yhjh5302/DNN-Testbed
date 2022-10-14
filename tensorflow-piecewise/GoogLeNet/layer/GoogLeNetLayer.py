from common import *
from GoogLeNetModel import *

def processing(inputs, model):
    outputs = model(inputs)
    return outputs

# python3 GoogLeNetLayer.py --layer_list 'conv1' 'conv1_maxpool' 'conv2' 'conv3' 'conv3_maxpool' 'inception3a_branch1' 'inception3a_branch2_conv1' 'inception3a_branch2_conv2' 'inception3a_branch3_conv1' 'inception3a_branch3_conv2' 'inception3a_branch4_maxpool' 'inception3a_branch4_conv' 'inception3a_concat' 'inception3b_branch1' 'inception3b_branch2_conv1' 'inception3b_branch2_conv2' 'inception3b_branch3_conv1' 'inception3b_branch3_conv2' 'inception3b_branch4_maxpool' 'inception3b_branch4_conv' 'inception3b_concat' 'inception3b_maxpool' 'inception4a_branch1' 'inception4a_branch2_conv1' 'inception4a_branch2_conv2' 'inception4a_branch3_conv1' 'inception4a_branch3_conv2' 'inception4a_branch4_maxpool' 'inception4a_branch4_conv' 'inception4a_concat' 'inception4b_branch1' 'inception4b_branch2_conv1' 'inception4b_branch2_conv2' 'inception4b_branch3_conv1' 'inception4b_branch3_conv2' 'inception4b_branch4_maxpool' 'inception4b_branch4_conv' 'inception4b_concat' 'inception4c_branch1' 'inception4c_branch2_conv1' 'inception4c_branch2_conv2' 'inception4c_branch3_conv1' 'inception4c_branch3_conv2' 'inception4c_branch4_maxpool' 'inception4c_branch4_conv' 'inception4c_concat' 'inception4d_branch1' 'inception4d_branch2_conv1' 'inception4d_branch2_conv2' 'inception4d_branch3_conv1' 'inception4d_branch3_conv2' 'inception4d_branch4_maxpool' 'inception4d_branch4_conv' 'inception4d_concat' 'inception4e_branch1' 'inception4e_branch2_conv1' 'inception4e_branch2_conv2' 'inception4e_branch3_conv1' 'inception4e_branch3_conv2' 'inception4e_branch4_maxpool' 'inception4e_branch4_conv' 'inception4e_concat' 'inception4e_maxpool' 'inception5a_branch1' 'inception5a_branch2_conv1' 'inception5a_branch2_conv2' 'inception5a_branch3_conv1' 'inception5a_branch3_conv2' 'inception5a_branch4_maxpool' 'inception5a_branch4_conv' 'inception5a_concat' 'inception5b_branch1' 'inception5b_branch2_conv1' 'inception5b_branch2_conv2' 'inception5b_branch3_conv1' 'inception5b_branch3_conv2' 'inception5b_branch4_maxpool' 'inception5b_branch4_conv' 'inception5b_concat' 'fully_connected' --prev_addr='' --prev_port='30001' --next_addr='localhost' --next_port='30000' --set_gpu='true' --vram_limit=1024
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow')
    parser.add_argument('--layer_list', default=['conv1', 'conv1_maxpool', 'conv2', 'conv3', 'conv3_maxpool', 'inception3a_branch1', 'inception3a_branch2_conv1', 'inception3a_branch2_conv2', 'inception3a_branch3_conv1', 'inception3a_branch3_conv2', 'inception3a_branch4_maxpool', 'inception3a_branch4_conv', 'inception3a_concat', 'inception3b_branch1', 'inception3b_branch2_conv1', 'inception3b_branch2_conv2', 'inception3b_branch3_conv1', 'inception3b_branch3_conv2', 'inception3b_branch4_maxpool', 'inception3b_branch4_conv', 'inception3b_concat', 'inception3b_maxpool', 'inception4a_branch1', 'inception4a_branch2_conv1', 'inception4a_branch2_conv2', 'inception4a_branch3_conv1', 'inception4a_branch3_conv2', 'inception4a_branch4_maxpool', 'inception4a_branch4_conv', 'inception4a_concat', 'inception4b_branch1', 'inception4b_branch2_conv1', 'inception4b_branch2_conv2', 'inception4b_branch3_conv1', 'inception4b_branch3_conv2', 'inception4b_branch4_maxpool', 'inception4b_branch4_conv', 'inception4b_concat', 'inception4c_branch1', 'inception4c_branch2_conv1', 'inception4c_branch2_conv2', 'inception4c_branch3_conv1', 'inception4c_branch3_conv2', 'inception4c_branch4_maxpool', 'inception4c_branch4_conv', 'inception4c_concat', 'inception4d_branch1', 'inception4d_branch2_conv1', 'inception4d_branch2_conv2', 'inception4d_branch3_conv1', 'inception4d_branch3_conv2', 'inception4d_branch4_maxpool', 'inception4d_branch4_conv', 'inception4d_concat', 'inception4e_branch1', 'inception4e_branch2_conv1', 'inception4e_branch2_conv2', 'inception4e_branch3_conv1', 'inception4e_branch3_conv2', 'inception4e_branch4_maxpool', 'inception4e_branch4_conv', 'inception4e_concat', 'inception4e_maxpool', 'inception5a_branch1', 'inception5a_branch2_conv1', 'inception5a_branch2_conv2', 'inception5a_branch3_conv1', 'inception5a_branch3_conv2', 'inception5a_branch4_maxpool', 'inception5a_branch4_conv', 'inception5a_concat', 'inception5b_branch1', 'inception5b_branch2_conv1', 'inception5b_branch2_conv2', 'inception5b_branch3_conv1', 'inception5b_branch3_conv2', 'inception5b_branch4_maxpool', 'inception5b_branch4_conv', 'inception5b_concat', 'fully_connected'], nargs='+', type=str, help='layer list for this application')
    parser.add_argument('--prev_addr', default='localhost', type=str, help='Previous node address')
    parser.add_argument('--prev_port', default=30011, type=int, help='Previous node port')
    parser.add_argument('--next_addr', default='localhost', type=str, help='Next node address')
    parser.add_argument('--next_port', default=30010, type=int, help='Next node port')
    parser.add_argument('--set_gpu', default=True, type=str2bool, help='If you want to use GPU, set "True"')
    parser.add_argument('--vram_limit', default=1024, type=int, help='Vram limitation')
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
    print(args.layer_list)
    model = GoogLeNet(name='GoogLeNet', layer_list=args.layer_list)

    # for cuDNN loading
    model(model.get_random_input())

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

    # for data multi-processing
    recv_data_list = []
    recv_data_lock = threading.Lock()
    send_data_list = []
    send_data_lock = threading.Lock()
    recv_time_list = []
    recv_time_lock = threading.Lock()
    proc_time_list = []
    proc_time_lock = threading.Lock()
    _stop_event = threading.Event()
    threading.Thread(target=recv_data, args=(p, recv_data_list, recv_data_lock, recv_time_list, recv_time_lock, proc_time_list, proc_time_lock, _stop_event)).start()
    threading.Thread(target=send_data, args=(next_sock, send_data_list, send_data_lock, _stop_event)).start()

    while True:
        inputs = bring_data(recv_data_list, recv_data_lock, _stop_event)
        outputs = processing(inputs, model)
        with send_data_lock:
            send_data_list.append(outputs)

        proc_end = time.time()
        with recv_time_lock:
            T_tr = recv_time_list.pop(0)
        with proc_time_lock:
            T_cp = proc_time_list.pop(0)
        print("T_tr\t{}".format(T_tr))
        print("T_cp\t{}".format(proc_end - T_cp))

    prev_sock.close()
    next_sock.close()