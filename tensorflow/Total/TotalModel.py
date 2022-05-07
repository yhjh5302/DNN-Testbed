from common import *
from VGGNetModel import *
from AlexNetModel import *
import numpy as np


def processing(inputs, model):
    request_id = inputs[0]
    outputs = model(inputs[-1])
    return request_id, outputs


PARTITION_INFOS={
    # "VGG-1": ('features1', 'features2', 'features3'),  # partition 1
    # "VGG-2": ('features4', 'features5'),               # partition 2
    # "VGG-3": ('classifier1', 'classifier2', 'classifier3'),  # partition 3
    
    # "AlexNet-1": ('features_1', 'features_2'),
    # "AlexNet-2": ('features_3', 'features_4', 'features_5'),
    # "AlexNet-3": ('classifier_1', 'classifier_2', 'classifier_3'),
    "AlexNet-in": ('input'),
    "AlexNet-1": ('feature_1_1', 'feature_2_1', 'feature_3_1', 'feature_4_1', 'feature_5_1', 'classifier_1_1', 'classifier_2_1', 'classifier_3_1'),
    "AlexNet-2": ('feature_1_2', 'feature_2_2', 'feature_3_2', 'feature_4_2', 'feature_5_2', 'classifier_1_2', 'classifier_2_2', 'classifier_3_2'),
    "AlexNet-out": ('output',)
    
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
    
    "AlexNet-in": 2,
    "AlexNet-1": 3,
    "AlexNet-2": 4,
    "AlexNet-out": 5,
}
succ={
    PARTITION_IDX_MAP['AlexNet-in']:(3,4),
    3:(5,),
    4:(5,),
}


class DAGManager:
    def __init__(self):
        self.dag_infos = dict()
        self.partition_input_sample = dict()
        self.dag_input_indices = {
            PARTITION_IDX_MAP['AlexNet-out']:{
                PARTITION_IDX_MAP['AlexNet-1']:(0,2048),
                PARTITION_IDX_MAP['AlexNet-2']:(2048,4096)
            }
        }
        self.recv_data_dict = dict()
        self.input_num_infos = {
            PARTITION_IDX_MAP['AlexNet-in']: 1,
            PARTITION_IDX_MAP['AlexNet-1']: 1,
            PARTITION_IDX_MAP['AlexNet-2']: 1,
            PARTITION_IDX_MAP['AlexNet-out']: 2
        }

    def recv_data(self, inputs):
        req_id = inputs[0]
        source_partion = inputs[1]
        target_partition = inputs[2]
        data = inputs[3]
        if self.input_num_infos[target_partition] == 1:
            return inputs
        else:
            if req_id not in self.recv_data_dict:
                self.recv_data_dict[req_id] = [1, np.zeros_like(self.partition_input_sample[target_partition])]
                
            else:
                self.recv_data_dict[req_id]['num'] += 1
            self.recv_data_dict[req_id][1][:,:,:,self.dag_input_indices[target_partition][source_partion][0]:self.dag_input_indices[target_partition][source_partion][1]] = data[:]  #chk demesion
          
            if self.recv_data_dict[req_id]['num'] == self.input_num_infos[target_partition]:
                result = (req_id, source_partion, target_partition, self.recv_data_dict[req_id][1])
                del self.recv_data_dict[req_id]
                print("working!!!")
                return result
            else:
                return None


# python3 VGGNetLayer.py --layer_list 'features1' 'features2' 'features3' 'features4' 'features5' 'classifier1' 'classifier2' 'classifier3' --prev_addr='' --prev_port='30031' --next_addr='localhost' --next_port='30030' --scheduler_addr='localhost' --scheduler_port='30050' --set_gpu='true' --vram_limit=1024
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow')
    # parser.add_argument('--layer_list', default=['features1', 'features2', 'features3', 'features4', 'features5', 'classifier1', 'classifier2', 'classifier3'], nargs='+', type=str, help='layer list for this application')
    parser.add_argument('--deployed_lst', default=["VGG-1", "VGG-2", "VGG-3", "AlexNet-1", "AlexNet-2", "AlexNet-3"], nargs='+', type=str, help='layer list for this application')
    parser.add_argument('--device_addr_lst', default=['10.96.0.231', '10.96.0.232','10.96.0.233','10.96.0.234','10.96.0.235','10.96.0.236'], nargs='+', type=str, help='address list of kubernetes cluster')
    parser.add_argument('--prev_addr', default='10.96.0.231', type=str, help='Previous node address')
    parser.add_argument('--prev_port', default=30031, type=int, help='Previous node port')
    parser.add_argument('--next_addr', default="localhost", type=str, help='Next node address')
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
    model_dict = dict()
    for partition_name in args.deployed_lst:
        if partition_name.find("VGG") > -1:
            model = VGGNet_layer(name=partition_name, layer_list=PARTITION_INFOS[partition_name])
        
        elif partition_name.find("AlexNet") > -1:
            model = AlexNet_layer(name=partition_name, layer_list=PARTITION_INFOS[partition_name])


        # for cuDNN loading
        model(model.get_random_input())

        print('Pretrained model loading done!')
        model_dict[PARTITION_IDX_MAP[partition_name]] = model
    
    dag_man = DAGManager()

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

    # if args.set_gpu:
    #     scheduler_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     scheduler_sock.settimeout(1000) # 1000 seconds
    #     scheduler_sock.connect((args.scheduler_addr, args.scheduler_port))
    #     print('Scheduler is ready, Connected by', args.scheduler_addr)

    # threading.Thread(target=start_partition_sorket, args=(p, next_sock))
    
    dag_input_dict = dict()
    recv_data_list = list()
    send_data_list = list()
    recv_time_list = []
    recv_time_lock = threading.Lock()
    proc_time_list = []
    proc_time_lock = threading.Lock()

    recv_data_lock = threading.Lock()
    send_data_lock = threading.Lock()
    
    _stop_event = threading.Event()
    threading.Thread(target=recv_data, args=(p, recv_data_list, recv_data_lock, recv_time_list, recv_time_lock, proc_time_list, proc_time_lock, _stop_event, dag_man)).start()
    threading.Thread(target=send_data, args=(next_sock, send_data_list, send_data_lock, _stop_event)).start()  # todo change send datas

    while True:
        # if args.set_gpu:
        #     inputs = bring_data(recv_data_list, recv_data_lock, _stop_event, scheduler_sock)
        #     request_id, outputs = processing(inputs, model)
        #     send_done(scheduler_sock)
        # else:
        #     inputs = bring_data(recv_data_list, recv_data_lock, _stop_event)
        #     request_id, outputs = processing(inputs, model)
        inputs = bring_data(recv_data_list, recv_data_lock, _stop_event)
        request_id, outputs = processing(inputs, model)
        with send_data_lock:
            for succ_partition in succ[inputs[2]]:
                next_inputs = (request_id, inputs[2], succ_partition, outputs)  # request id, source partition id, target partition id, output
                send_data_list.append(next_inputs)

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