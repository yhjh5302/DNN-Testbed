from common import *
from VGGNetModel import *
from AlexNetModel import *


def processing(inputs, model):
    outputs = model(inputs)
    return outputs


PARTITION_INFOS={
    "VGG-1": ('features1', 'features2', 'features3'),  # partition 1
    "VGG-2": ('features4', 'features5'),               # partition 2
    "VGG-3": ('classifier1', 'classifier2', 'classifier3'),  # partition 3
    
    "AlexNet-1": ('features_1', 'features_2'),
    "AlexNet-2": ('features_3', 'features_4', 'features_5'),
    "AlexNet-3": ('classifier_1', 'classifier_2', 'classifier_3'),
    
    # "NiN": (
    #     (),

    # ),
    # "ResNet": (
    #     (),

    # )
}


PARTITION_IDX_MAP={
    "VGG-1": 0,
    "VGG-2": 1,
    "VGG-3": 2,
    
    "AlexNet-1": 3,
    "AlexNet-2": 4,
    "AlexNet-3": 5,
}


def start_partition_sorket(p, next_sock):
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
        if args.set_gpu:
            inputs = bring_data(recv_data_list, recv_data_lock, _stop_event, scheduler_sock)
            outputs = processing(inputs, model)
            send_done(scheduler_sock)
        else:
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


# python3 VGGNetLayer.py --layer_list 'features1' 'features2' 'features3' 'features4' 'features5' 'classifier1' 'classifier2' 'classifier3' --prev_addr='' --prev_port='30031' --next_addr='localhost' --next_port='30030' --scheduler_addr='localhost' --scheduler_port='30050' --set_gpu='true' --vram_limit=1024
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow')
    # parser.add_argument('--layer_list', default=['features1', 'features2', 'features3', 'features4', 'features5', 'classifier1', 'classifier2', 'classifier3'], nargs='+', type=str, help='layer list for this application')
    parser.add_argument('--deployed_lst', default=["VGG-1", "VGG-2", "VGG-3", "AlexNet-1", "AlexNet-2", "AlexNet-3"], nargs='+', type=str, help='layer list for this application')
    parser.add_argument('--device_addr_lst', default=['10.96.0.231', '10.96.0.232','10.96.0.233','10.96.0.234','10.96.0.235','10.96.0.236'], nargs='+', type=str, help='address list of kubernetes cluster')
    parser.add_argument('--prev_addr', default='10.96.0.231', type=str, help='Previous node address')
    parser.add_argument('--prev_port', default=30031, type=int, help='Previous node port')
    parser.add_argument('--next_addr_lst', default=[0,0,0,1,1,0,1,0], nargs='+', type=int, help='Next node address')
    parser.add_argument('--next_port', default=30030,  type=int, help='Next node port')
    parser.add_argument('--scheduler_addr', default='10.96.0.250', type=str, help='Scheduler address')
    parser.add_argument('--scheduler_port', default=30050, type=int, help='Scheduler port')
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
    # model = VGGNet_layer(name='VGG-16', layer_list=args.layer_list)
    model_lst = list()
    for partition_name in args.deployed_lst:
        if partition_name.find("VGG") > -1:
            model = VGGNet_layer(name=partition_name, layer_list=PARTITION_INFOS[partition_name])
        
        elif partition_name.find("AlexNet") > -1:
            model = AlexNet_layer(name=partition_name, layer_list=PARTITION_INFOS[partition_name])


        # for cuDNN loading
        model(model.get_random_input())

        print('Pretrained model loading done!')
        model_lst.append(model)

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

        # threading.Thread(target=start_partition_sorket, args=(p, next_sock))
    
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
        if args.set_gpu:
            inputs = bring_data(recv_data_list, recv_data_lock, _stop_event, scheduler_sock)
            outputs = processing(inputs, model)
            send_done(scheduler_sock)
        else:
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
    if args.set_gpu:
        scheduler_sock.close()