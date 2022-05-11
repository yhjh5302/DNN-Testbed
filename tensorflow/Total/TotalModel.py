from common import *
from VGGNetModel import *
from AlexNetModel import *
from ResNetModel import *
from NiNModel import *
import numpy as np
from time import sleep
from copy import deepcopy

from dag_config import *


def processing(inputs, model):
    request_id = inputs[0]
    outputs = model(inputs[-1])
    return request_id, outputs


# python3 VGGNetLayer.py --layer_list 'features1' 'features2' 'features3' 'features4' 'features5' 'classifier1' 'classifier2' 'classifier3' --prev_addr='' --prev_port='30031' --next_addr='localhost' --next_port='30030' --scheduler_addr='localhost' --scheduler_port='30050' --set_gpu='true' --vram_limit=1024
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow')
    # parser.add_argument('--layer_list', default=['features1', 'features2', 'features3', 'features4', 'features5', 'classifier1', 'classifier2', 'classifier3'], nargs='+', type=str, help='layer list for this application')
    parser.add_argument('--deployed_list', default=[
        "AlexNet-in", "AlexNet-1", "AlexNet-2", "AlexNet-out", 
        "VGG", "NiN", 
        "ResNet-in", "ResNet-CNN_1_2", "ResNet-CNN_2_1", "ResNet-CNN_3_2", 
        "ResNet-CNN_4_1", "ResNet-CNN_5_2", "ResNet-CNN_6_1", "ResNet-CNN_7_2", 
        "ResNet-CNN_8_1", "ResNet-CNN_9_2", "ResNet-CNN_10_1", "ResNet-CNN_11_2", 
        "ResNet-CNN_12_1", "ResNet-CNN_13_2", "ResNet-CNN_14_1", "ResNet-CNN_15_2", 
        "ResNet-CNN_16_1", "ResNet-CNN_17"], nargs='*', type=str, help='layer list for this application')
    parser.add_argument('--device_index', default=1, type=int, help='device index for device')
    parser.add_argument('--device_addr_list', default=['192.168.1.13', '192.168.1.4'], nargs='+', type=str, help='address list of kubernetes cluster')
    parser.add_argument('--resv_port_list', default=[30030, 30030], nargs='+', type=int, help='receive port')
    parser.add_argument('--send_port_list', default=[30031, 30031], nargs='+', type=int, help='send port')
    parser.add_argument('--partition_location', default=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], nargs='+', type=int, help='deployed device number')
    parser.add_argument('--generator_idx', default=0, type=int, help='generator container idx')
    parser.add_argument('--set_gpu', default=True, type=str2bool, help='If you want to use GPU, set "True"')
    parser.add_argument('--vram_limit', default=1024*5, type=int, help='Vram limitation')
    parser.add_argument('--p', default=[
        0.02481079, 0.02233019, 0.02436261, 0.02627764, 0.0274137, 0.02624875, 
        0.02336616, 0.02264515, 0.02113937, 0.02546869, 0.02113972, 0.0260701,
        0.02474534, 0.02531103, 0.02335365, 0.02519561, 0.02310146, 0.02305768, 
        0.02115483, 0.07413184, 0.07294686, 0.02995578, 0.17000617, 0.17070863], nargs='*', type=float, help='percentage of time to use, total sum must be 1')
    parser.add_argument('--time', default=1.0, type=float, help='second')
    args = parser.parse_args()

    if args.set_gpu:
        gpu_devices = tf.config.list_physical_devices(device_type='GPU')
        if not gpu_devices:
            raise ValueError('Cannot detect physical GPU device in TF')
        # tf.config.set_logical_device_configuration(gpu_devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=args.vram_limit)])
        tf.config.list_logical_devices()
    else:
        tf.config.set_visible_devices([], 'GPU')

    # model loading
    # model = VGGNet_layer(name='VGG-16', layer_list=args.layer_list)
    model_dict = dict()
    for partition_name in args.deployed_list:
        model = None
        if partition_name.find("VGG") > -1:
            model = VGGNet_layer(name=partition_name, layer_list=PARTITION_INFOS[partition_name])
        
        elif partition_name.find("AlexNet") > -1:
            model = AlexNet_layer(name=partition_name, layer_list=PARTITION_INFOS[partition_name])

        elif partition_name.find("ResNet") > -1:
            model = ResNet_layer(name=partition_name, layer_list=PARTITION_INFOS[partition_name])

        elif partition_name.find("NiN") > -1:
            model = NiN_layer(name=partition_name, layer_list=PARTITION_INFOS[partition_name])

        # for cuDNN loading
        if model is not None:
            model(model.get_random_input())
            print('Pretrained model loading done!')
            model_dict[PARTITION_IDX_MAP[partition_name]] = model

    
    dag_man = DAGManager()
    
    # dag_input_dict = dict()
    recv_data_dict = {
        'waiting_num': 0,
        'proc': dict(),
        'partitions':list()
    }

    deployed_idx_dict = dict()
    recv_data_lock_dict = dict()

    for partition in args.deployed_list:
        partition_idx = PARTITION_IDX_MAP[partition]
        recv_data_dict[partition_idx] = list()
        recv_data_dict['proc'][partition_idx] = None
        recv_data_dict['partitions'].append(partition_idx)
        deployed_idx_dict[partition_idx] = len(recv_data_dict['partitions']) - 1
        recv_data_lock_dict[partition_idx] = threading.Lock()
    
    _stop_event = threading.Event()

    receive_socket = list()

    send_socket = list()
    dev_send_list = list()
    dev_send_data_list = list()
    dev_send_lock_list = list()
    proc_start_time = dict()
    
    partition_location = args.partition_location
    for i in range(len(args.device_addr_list)):

        if i == args.device_index:
            send_data_list = None
            send_data_lock = None
        
        else:
            send_data_list = list()
            send_data_lock = threading.Lock()
        dev_send_data_list.append(send_data_list)
        dev_send_lock_list.append(send_data_lock)
    
    for i in range(len(partition_location)):
        if partition_location[i] == args.device_index:
            partition_location[i] = -1  # device is self

    print("partition location", partition_location)

    
    for i in range(len(args.device_addr_list)):
        if i != args.device_index:
            if i > args.device_index:
                resv_conn, resv_addr = open_resv_sock("", args.resv_port_list[i])
                client_addr = resv_addr[0]
                send_sock, send_addr = open_send_sock(args.device_addr_list[i], args.send_port_list[i])
            else:
                send_sock, send_addr = open_send_sock(args.device_addr_list[i], args.resv_port_list[args.device_index])
                resv_conn, resv_addr = open_resv_sock("", args.send_port_list[args.device_index])

            print("connection with {} established".format(args.device_addr_list[i]))

            threading.Thread(target=recv_data, args=(resv_conn, recv_data_dict, recv_data_lock_dict, _stop_event, dag_man)).start()
            threading.Thread(target=send_data, args=(send_sock, dev_send_data_list[i], dev_send_lock_list[i], _stop_event)).start()
    
    
    print("all connection is established")
    weights = [p * args.time for p in args.p]   # correct options
    init_prob = [p * args.time for p in args.p]
    
    while True:        
        while sum(weights) < 1e-8:
            weights = [p * args.time for p in args.p]
            
        inputs, start = bring_data(recv_data_dict, recv_data_lock_dict, _stop_event, prob=weights, init_prob=init_prob)
        T_tr = inputs[1]
        pure_tr_time = inputs[2]
        inputs = inputs[0]
        idx = inputs[2]
        request_id, outputs = processing(inputs, model_dict[idx])
        if type(outputs) in (list, tuple):
            outputs = outputs[0]  # remove short cut

        proc_end = time.time()
        T_cp = proc_end - start

        deployed_idx = deployed_idx_dict[idx]

        weights[deployed_idx] = max(weights[deployed_idx] - T_cp, 0)           # change index
        
        if inputs[2] in DAG_SUCCESSORS:
            for succ_partition in DAG_SUCCESSORS[inputs[2]]:
                if len(DAG_SUCCESSORS[inputs[2]]) == 1:
                    next_inputs = (request_id, inputs[2], succ_partition, outputs)
                else:
                    next_inputs = (request_id, inputs[2], succ_partition, deepcopy(outputs))
                deployment_idx = partition_location[succ_partition]
                # print("deployment_idx", deployment_idx)
                if deployment_idx > -1:
                    with dev_send_lock_list[deployment_idx]:
                        # request id, source partition id, target partition id, output
                        dev_send_data_list[deployment_idx].append(next_inputs)
                else:  # local
                    cur_time = time.time()
                    next_inputs = dag_man.recv_data(next_inputs, cur_time, cur_time)
                    if next_inputs is not None:
                        with recv_data_lock_dict[succ_partition]:
                            target_partition = next_inputs[0][2]
                            if recv_data_dict['proc'][target_partition] is None:
                                recv_data_dict['proc'][target_partition] = (next_inputs, cur_time)
                            else:
                                recv_data_dict[target_partition].append((next_inputs, cur_time))
                            recv_data_dict['waiting_num'] += 1
            # todo transmission time
            # with recv_time_lock:
            #     T_tr = recv_time_list.pop(0)
        else:
            with dev_send_lock_list[args.generator_idx]: # send return
                result_packet = (request_id, inputs[2], -1, outputs)
                dev_send_data_list[args.generator_idx].append(result_packet)
        
        print("{}\tT_tr_pure\t{}".format(REVERSE_IDX_MAP[idx], pure_tr_time))
        print("{}\tT_tr\t{}".format(REVERSE_IDX_MAP[idx], T_tr))
        print("{}\tT_cp\t{}".format(REVERSE_IDX_MAP[idx], T_cp))

    prev_sock.close()
    next_sock.close()