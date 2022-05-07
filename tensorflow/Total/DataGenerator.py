from common import *
from tensorflow import keras
import argparse
#import multiprocessing as mp


model_idx = {
    'alexnet':0
}
def image_sender(model_name, next_socket, images, labels, label_list, label_lock, time_list, time_lock, arrival_rate, _stop_event, num_model=1):
    for _ in range(100):
        # sleep before sending
        #time.sleep(1/arrival_rate)
        time.sleep(1) # per 1 seconds

        # reading queue
        batch_size = 1
        for i in range(arrival_rate):            
            print(model_name," send 1\n")
            idx = np.random.randint(10000-batch_size)
            data = images[idx:idx+batch_size]
            req_id = (num_model * i) + model_idx[model_name]
            data = (req_id, 0, 2, data)
            
            with label_lock:
                label_list.append(labels.flatten()[idx:idx+batch_size])
            print(model_name," send 2\n")
            # sending data
            with time_lock:
                time_list.append(time.time())
            print(model_name," send 3\n")
            send_input(next_socket, data, _stop_event)
            print(model_name," send 4\n")
    #_stop_event.set()

def image_recver(model_name, conn, label_list, label_lock, time_list, time_lock, _stop_event):
    init = False
    while _stop_event.is_set() == False:
        # make data receiving thread
        outputs = recv_output(conn, _stop_event)
        if init == False:
            init = time.time()
        predicted = tf.argmax(outputs, 1)
        with label_lock:
            answer = label_list.pop(0)
        correct = np.sum(predicted == answer)

        with time_lock:
            start = time_list.pop(0)
        # wait for response
        print(model_name, "\t", time.time() - start, "\t", time.time() - init)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow')
    parser.add_argument('--set_gpu', default=False, type=str2bool, help='If you want to use GPU, set "True"')
    parser.add_argument('--alexnet_prev_addr', default='10.96.0.200', type=str, help='Previous node address')
    parser.add_argument('--alexnet_prev_port', default=30000, type=int, help='Previous node port')
    parser.add_argument('--alexnet_next_addr', default='10.96.0.201', type=str, help='Next node address')
    parser.add_argument('--alexnet_next_port', default=30001, type=int, help='Next node port')
    parser.add_argument('--googlenet_prev_addr', default='10.96.0.200', type=str, help='Previous node address')
    parser.add_argument('--googlenet_prev_port', default=30010, type=int, help='Previous node port')
    parser.add_argument('--googlenet_next_addr', default='10.96.0.211', type=str, help='Next node address')
    parser.add_argument('--googlenet_next_port', default=30011, type=int, help='Next node port')
    parser.add_argument('--mobilenet_prev_addr', default='10.96.0.200', type=str, help='Previous node address')
    parser.add_argument('--mobilenet_prev_port', default=30020, type=int, help='Previous node port')
    parser.add_argument('--mobilenet_next_addr', default='10.96.0.221', type=str, help='Next node address')
    parser.add_argument('--mobilenet_next_port', default=30021, type=int, help='Next node port')
    parser.add_argument('--vggnet_prev_addr', default='10.96.0.200', type=str, help='Previous node address')
    parser.add_argument('--vggnet_prev_port', default=30030, type=int, help='Previous node port')
    parser.add_argument('--vggnet_next_addr', default='10.96.0.231', type=str, help='Next node address')
    parser.add_argument('--vggnet_next_port', default=30031, type=int, help='Next node port')
    parser.add_argument('--vggfnet_prev_addr', default='10.96.0.200', type=str, help='Previous node address')
    parser.add_argument('--vggfnet_prev_port', default=30040, type=int, help='Previous node port')
    parser.add_argument('--vggfnet_next_addr', default='10.96.0.241', type=str, help='Next node address')
    parser.add_argument('--vggfnet_next_port', default=30041, type=int, help='Next node port')
    parser.add_argument('--alexnet_block', action='store_true', help='block?')
    parser.add_argument('--googlenet_block', action='store_true', help='block?')
    parser.add_argument('--mobilenet_block', action='store_true', help='block?')
    parser.add_argument('--vggnet_block', action='store_true', help='block?')
    parser.add_argument('--vggfnet_block', action='store_true', help='block?')
    parser.add_argument('--alexnet_arrival_rate', default=1, type=int, help='arrival rate')
    parser.add_argument('--googlenet_arrival_rate', default=1, type=int, help='arrival rate')
    parser.add_argument('--mobilenet_arrival_rate', default=1, type=int, help='arrival rate')
    parser.add_argument('--vggnet_arrival_rate', default=1, type=int, help='arrival rate')
    parser.add_argument('--vggfnet_arrival_rate', default=1, type=int, help='arrival rate')
    parser.add_argument('--vram_limit', default=0, type=int, help='Next node port')
    args = parser.parse_args()

    if args.set_gpu:
        gpu_devices = tf.config.list_physical_devices(device_type='GPU')
        if not gpu_devices:
            raise ValueError('Cannot detect physical GPU device in TF')
        tf.config.set_logical_device_configuration(gpu_devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=args.vram_limit)])
        tf.config.list_logical_devices()
    else:
        tf.config.set_visible_devices([], 'GPU')

    _, (images, labels) = keras.datasets.cifar10.load_data()
    images = images.reshape(10000, 32, 32, 3).astype('float32') / 255
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.alexnet_block == False:
        alexnet_next_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        alexnet_next_sock.settimeout(600)
        alexnet_next_sock.connect((args.alexnet_next_addr, args.alexnet_next_port))
        print('AlexNet next node is ready, Connected by', args.alexnet_next_addr)

        alexnet_prev_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        alexnet_prev_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        alexnet_prev_sock.bind((args.alexnet_prev_addr, args.alexnet_prev_port))
        alexnet_prev_sock.listen()
        alexnet_port, alexnet_addr = alexnet_prev_sock.accept()
        print('AlexNet prev node is ready, Connected by', alexnet_addr)

    if args.googlenet_block == False:
        googlenet_next_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        googlenet_next_sock.settimeout(600)
        googlenet_next_sock.connect((args.googlenet_next_addr, args.googlenet_next_port))
        print('GoogLenet next node is ready, Connected by', args.googlenet_next_addr)

        googlenet_prev_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        googlenet_prev_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        googlenet_prev_sock.bind((args.googlenet_prev_addr, args.googlenet_prev_port))
        googlenet_prev_sock.listen()
        googlenet_port, googlenet_addr = googlenet_prev_sock.accept()
        print('GoogLenet prev node is ready, Connected by', googlenet_addr)

    if args.mobilenet_block == False:
        mobilenet_next_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        mobilenet_next_sock.settimeout(600)
        mobilenet_next_sock.connect((args.mobilenet_next_addr, args.mobilenet_next_port))
        print('MobileNet next node is ready, Connected by', args.mobilenet_next_addr)

        mobilenet_prev_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        mobilenet_prev_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        mobilenet_prev_sock.bind((args.mobilenet_prev_addr, args.mobilenet_prev_port))
        mobilenet_prev_sock.listen()
        mobilenet_port, mobilenet_addr = mobilenet_prev_sock.accept()
        print('MobileNet prev node is ready, Connected by', mobilenet_addr)

    if args.vggnet_block == False:
        vggnet_next_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        vggnet_next_sock.settimeout(600)
        vggnet_next_sock.connect((args.vggnet_next_addr, args.vggnet_next_port))
        print('VGGNet next node is ready, Connected by', args.vggnet_next_addr)

        vggnet_prev_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        vggnet_prev_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        vggnet_prev_sock.bind((args.vggnet_prev_addr, args.vggnet_prev_port))
        vggnet_prev_sock.listen()
        vggnet_port, vggnet_addr = vggnet_prev_sock.accept()
        print('VGGNet prev node is ready, Connected by', vggnet_addr)

    if args.vggfnet_block == False:
        vggfnet_next_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        vggfnet_next_sock.settimeout(600)
        vggfnet_next_sock.connect((args.vggfnet_next_addr, args.vggfnet_next_port))
        print('VGGFNet next node is ready, Connected by', args.vggfnet_next_addr)

        vggfnet_prev_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        vggfnet_prev_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        vggfnet_prev_sock.bind((args.vggfnet_prev_addr, args.vggfnet_prev_port))
        vggfnet_prev_sock.listen()
        vggfnet_port, vggfnet_addr = vggfnet_prev_sock.accept()
        print('VGGFNet prev node is ready, Connected by', vggfnet_addr)

    input('Enter any key...')

    _stop_event = threading.Event()
    procs = []

    if args.alexnet_block == False:
        alexnet_label_list = []
        alexnet_time_list = []
        alexnet_label_lock = threading.Lock()
        alexnet_time_lock = threading.Lock()
        procs.append(threading.Thread(target=image_sender, args=("alexnet", alexnet_next_sock, images, labels, alexnet_label_list, alexnet_label_lock, alexnet_time_list, alexnet_time_lock, args.alexnet_arrival_rate, _stop_event)))
        procs.append(threading.Thread(target=image_recver, args=("alexnet", alexnet_port, alexnet_label_list, alexnet_label_lock, alexnet_time_list, alexnet_time_lock, _stop_event)))

    if args.googlenet_block == False:
        googlenet_label_list = []
        googlenet_time_list = []
        googlenet_label_lock = threading.Lock()
        googlenet_time_lock = threading.Lock()
        procs.append(threading.Thread(target=image_sender, args=("googlenet", googlenet_next_sock, images, labels, googlenet_label_list, googlenet_label_lock, googlenet_time_list, googlenet_time_lock, args.googlenet_arrival_rate, _stop_event)))
        procs.append(threading.Thread(target=image_recver, args=("googlenet", googlenet_port, googlenet_label_list, googlenet_label_lock, googlenet_time_list, googlenet_time_lock, _stop_event)))

    if args.mobilenet_block == False:
        mobilenet_label_list = []
        mobilenet_time_list = []
        mobilenet_label_lock = threading.Lock()
        mobilenet_time_lock = threading.Lock()
        procs.append(threading.Thread(target=image_sender, args=("mobilenet", mobilenet_next_sock, images, labels, mobilenet_label_list, mobilenet_label_lock, mobilenet_time_list, mobilenet_time_lock, args.mobilenet_arrival_rate, _stop_event)))
        procs.append(threading.Thread(target=image_recver, args=("mobilenet", mobilenet_port, mobilenet_label_list, mobilenet_label_lock, mobilenet_time_list, mobilenet_time_lock, _stop_event)))

    if args.vggnet_block == False:
        vggnet_label_list = []
        vggnet_time_list = []
        vggnet_label_lock = threading.Lock()
        vggnet_time_lock = threading.Lock()
        procs.append(threading.Thread(target=image_sender, args=("vggnet", vggnet_next_sock, images, labels, vggnet_label_list, vggnet_label_lock, vggnet_time_list, vggnet_time_lock, args.vggnet_arrival_rate, _stop_event)))
        procs.append(threading.Thread(target=image_recver, args=("vggnet", vggnet_port, vggnet_label_list, vggnet_label_lock, vggnet_time_list, vggnet_time_lock, _stop_event)))

    if args.vggfnet_block == False:
        vggfnet_label_list = []
        vggfnet_time_list = []
        vggfnet_label_lock = threading.Lock()
        vggfnet_time_lock = threading.Lock()
        procs.append(threading.Thread(target=image_sender, args=("vggfnet", vggfnet_next_sock, images, labels, vggfnet_label_list, vggfnet_label_lock, vggfnet_time_list, vggfnet_time_lock, args.vggfnet_arrival_rate, _stop_event)))
        procs.append(threading.Thread(target=image_recver, args=("vggfnet", vggfnet_port, vggfnet_label_list, vggfnet_label_lock, vggfnet_time_list, vggfnet_time_lock, _stop_event)))

    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()

    if args.alexnet_block == False:
        alexnet_next_sock.close()
        alexnet_prev_sock.close()

    if args.googlenet_block == False:
        googlenet_next_sock.close()
        googlenet_prev_sock.close()

    if args.mobilenet_block == False:
        mobilenet_next_sock.close()
        mobilenet_prev_sock.close()

    if args.vggnet_block == False:
        vggnet_next_sock.close()
        vggnet_prev_sock.close()

    if args.vggfnet_block == False:
        vggfnet_next_sock.close()
        vggfnet_prev_sock.close()