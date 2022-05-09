from common import *
from tensorflow import keras
import argparse
from dag_config import *
#import multiprocessing as mp




def image_sender(model_name, next_socket, socket_lock,images, labels, label_list, label_lock, time_dict, time_lock, arrival_rate, _stop_event, num_model=1):
    for _ in range(100):
        # sleep before sending
        #time.sleep(1/arrival_rate)
        time.sleep(1) # per 1 seconds

        # reading queue
        batch_size = 1
        for i in range(arrival_rate):            
            idx = np.random.randint(10000-batch_size)
            data = images[idx:idx+batch_size]
            req_id = (num_model * i) + MODEL_IDX[model_name]
            data = (req_id, -1, 0, data)
            
            with label_lock:
                label_list.append(labels.flatten()[idx:idx+batch_size])
            # sending data
            with time_lock:
                time_dict[req_id] = time.time()
            with socket_lock:
                send_input(next_socket, data, _stop_event)
    #_stop_event.set()

def image_recver(model_name, conn, label_list, label_lock, time_dict, time_lock, _stop_event):
    init = False
    while _stop_event.is_set() == False:
        # make data receiving thread
        outputs = recv_output(conn, _stop_event)
        req_idx=outputs[0]
        last_partiton = outputs[1]
        outputs = outputs[-1]
        if init == False:
            init = time.time()
        predicted = tf.argmax(outputs, -1)
        with label_lock:
            answer = label_list.pop(0)
        correct = np.sum(predicted == answer)

        with time_lock:
            start = time_dict[req_idx]
            del time_dict[req_idx]
        # wait for response
        print(model_name, "\t", time.time() - start, "\t", time.time() - init)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow')
    parser.add_argument('--device_addr_list', default=['localhost', 'localhost'], nargs='+', type=str, help='address list of kubernetes cluster')
    parser.add_argument('--resv_port_list', default=[30030, 30030], nargs='+', type=int, help='receive port')
    parser.add_argument('--send_port_list', default=[30031, 30031], nargs='+', type=int, help='send port')
    parser.add_argument('--device_index', default=0, type=int, help='device index for device')
    parser.add_argument('--partition_location', default=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], nargs='+', type=int, help='deployed device number')
    parser.add_argument('--alexnet_block', action='store_true', help='block?')
    parser.add_argument('--vggnet_block', action='store_true', help='block?')
    parser.add_argument('--nin_block', action='store_true', help='block?')
    parser.add_argument('--resnet_block', action='store_true', help='block?')
    parser.add_argument('--alexnet_arrival_rate', default=1, type=int, help='arrival rate')
    parser.add_argument('--vggnet_arrival_rate', default=1, type=int, help='arrival rate')
    parser.add_argument('--nin_arrival_rate', default=1, type=int, help='arrival rate')
    parser.add_argument('--resnet_arrival_rate', default=1, type=int, help='arrival rate')
    parser.add_argument('--vram_limit', default=0, type=int, help='Next node port')
    args = parser.parse_args()

    tf.config.set_visible_devices([], 'GPU')

    _, (images, labels) = keras.datasets.cifar10.load_data()
    images = images.reshape(10000, 32, 32, 3).astype('float32') / 255
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dev_send_sock_list = list()
    dev_send_lock_list = list()
    dev_resv_sock_list = list()
    dev_resv_lock_list = list()
        
    
    for i in range(len(args.device_addr_list)):
        if i == args.device_index:
            send_data_lock = None
            resv_data_lock = None
            resv_conn = None
            send_sock = None

        
        else:
            send_data_lock = threading.Lock()
            resv_data_lock = threading.Lock()
        
            resv_opt = (args.device_addr_list[i], args.resv_port_list[i])
            send_opt = (args.device_addr_list[i], args.send_port_list[i])
            if i < args.device_index:
                resv_conn, resv_addr, send_sock, send_addr = server_socket(resv_opt, send_opt)
            else:
                resv_conn, resv_addr, send_sock, send_addr = client_socket(resv_opt, send_opt)

            print("connection with {} established".format(i))
        
        dev_send_sock_list.append(send_sock)
        dev_send_lock_list.append(send_data_lock)
        dev_resv_sock_list.append(resv_conn)
        dev_resv_lock_list.append(resv_data_lock)


    input('Enter any key...')

    _stop_event = threading.Event()
    procs = []

    if args.alexnet_block == False:
        alexnet_label_list = []
        alexnet_time_dict = dict()
        alexnet_label_lock = threading.Lock()
        alexnet_time_lock = threading.Lock()
        dev_id = args.partition_location[MODEL_START_PARTITION['alexnet']]
        procs.append(threading.Thread(target=image_sender, args=("alexnet", dev_send_sock_list[dev_id], dev_send_lock_list[dev_id], images, labels, alexnet_label_list, alexnet_label_lock, alexnet_time_dict, alexnet_time_lock, args.alexnet_arrival_rate, _stop_event)))
        dev_id = args.partition_location[MODEL_END_PARTITION['alexnet']]
        procs.append(threading.Thread(target=image_recver, args=("alexnet", dev_resv_sock_list[dev_id], alexnet_label_list, alexnet_label_lock, alexnet_time_dict, alexnet_time_lock, _stop_event)))

    if args.vggnet_block == False:
        vggnet_label_list = []
        vggnet_time_dict = dict()
        vggnet_label_lock = threading.Lock()
        vggnet_time_lock = threading.Lock()
        dev_id = args.partition_location[MODEL_START_PARTITION['vggnet']]
        procs.append(threading.Thread(target=image_sender, args=("vggnet", dev_send_sock_list[dev_id], dev_send_lock_list[dev_id], images, labels, vggnet_label_list, vggnet_label_lock, vggnet_time_dict, vggnet_time_lock, args.vggnet_arrival_rate, _stop_event)))
        dev_id = args.partition_location[MODEL_END_PARTITION['vggnet']]
        procs.append(threading.Thread(target=image_recver, args=("vggnet", dev_resv_sock_list[dev_id], vggnet_label_list, vggnet_label_lock, vggnet_time_dict, vggnet_time_lock, _stop_event)))

    if args.nin_block == False:
        nin_label_list = []
        nin_time_dict = dict()
        nin_label_lock = threading.Lock()
        nin_time_lock = threading.Lock()
        dev_id = args.partition_location[MODEL_START_PARTITION['nin']]
        procs.append(threading.Thread(target=image_sender, args=("nin", dev_send_sock_list[dev_id], dev_send_lock_list[dev_id], images, labels, nin_label_list, nin_label_lock, nin_time_dict, nin_time_lock, args.nin_arrival_rate, _stop_event)))
        dev_id = args.partition_location[MODEL_END_PARTITION['nin']]
        procs.append(threading.Thread(target=image_recver, args=("nin", dev_resv_sock_list[dev_id], nin_label_list, nin_label_lock, nin_time_dict, nin_time_lock, _stop_event)))

    if args.resnet_block == False:
        resnet_label_list = []
        resnet_time_dict = dict()
        resnet_label_lock = threading.Lock()
        resnet_time_lock = threading.Lock()
        dev_id = args.partition_location[MODEL_START_PARTITION['resnet']]
        procs.append(threading.Thread(target=image_sender, args=("resnet", dev_send_sock_list[dev_id], dev_send_lock_list[dev_id], images, labels, resnet_label_list, resnet_label_lock, resnet_time_dict, resnet_time_lock, args.resnet_arrival_rate, _stop_event)))
        dev_id = args.partition_location[MODEL_END_PARTITION['resnet']]
        procs.append(threading.Thread(target=image_recver, args=("resnet", dev_resv_sock_list[dev_id], resnet_label_list, resnet_label_lock, resnet_time_dict, resnet_time_lock, _stop_event)))
    

    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()

