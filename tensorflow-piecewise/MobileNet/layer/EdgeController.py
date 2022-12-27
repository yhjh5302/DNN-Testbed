from common import *
from tensorflow import keras
import argparse
#import multiprocessing as mp


def scheduler(model_name, next_sock, images, labels, label_list, label_lock, time_list, time_lock, _stop_event, num_model=1):
    schedule = [(partition_id, execution_order)]
    return schedule

def processing(inputs, model):
    outputs = model(inputs)
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow')
    parser.add_argument('--set_gpu', default=False, type=str2bool, help='If you want to use GPU, set "True"')
    parser.add_argument('--vram_limit', default=0, type=int, help='GPU memory limit')
    parser.add_argument('--self_addr', default='localhost', type=str, help='Previous node address')
    parser.add_argument('--self_port', default=30001, type=int, help='Previous node port')
    parser.add_argument('--master_addr', default='localhost', type=str, help='Previous node address')
    parser.add_argument('--master_port', default=30000, type=int, help='Previous node port')
    parser.add_argument('--data_path', default='./Data/', type=str, help='Image frame data path')
    parser.add_argument('--video_name', default='vdo.avi', type=str, help='Video file name')
    parser.add_argument('--roi_name', default='roi.jpg', type=str, help='RoI file name')
    parser.add_argument('--resolution', default=(854, 480), type=tuple, help='image resolution')
    parser.add_argument('--img_hw', default=416, type=int, help='inference engine input size')
    args = parser.parse_args()

    if len(args.addr_list) != len(port_list):
        raise RuntimeError("The number of addr and port does not match!")

    if args.set_gpu:
        gpu_devices = tf.config.list_physical_devices(device_type='GPU')
        if not gpu_devices:
            raise ValueError('Cannot detect physical GPU device in TF')
        tf.config.set_logical_device_configuration(gpu_devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=args.vram_limit)])
        tf.config.list_logical_devices()
    else:
        tf.config.set_visible_devices([], 'GPU')

    input('Enter any key...')

    # connection setup
    sock_list = []
    

    # for data multi-processing
    requests = []
    requests_lock = threading.Lock()
    schedule = []
    schedule_lock = threading.Lock()
    recv_data_list = [[] for _ in sock_list]
    recv_lock_list = [threading.Lock() for _ in sock_list]
    send_data_list = [[] for _ in sock_list]
    send_lock_list = [threading.Lock() for _ in sock_list]
    _stop_event = threading.Event()
    for i, sock in enumerate(sock_list):
        threading.Thread(target=edge_recv_data, args=(sock, recv_data_list[i], recv_lock_list[i], requests, requests_lock, _stop_event)).start()
        threading.Thread(target=edge_send_data, args=(sock, send_data_list[i], send_lock_list[i], schedule, schedule_lock, _stop_event)).start()
    threading.Thread(target=scheduler, args=(requests, requests_lock, schedule, schedule_lock, _stop_event)).start()

    while True:
        inputs = bring_data(recv_data_list, recv_lock_list, _stop_event)
        outputs = processing(inputs, model)
        with send_lock_list[]:
            send_data_list.append(outputs)

    prev_sock.close()
    next_sock.close()