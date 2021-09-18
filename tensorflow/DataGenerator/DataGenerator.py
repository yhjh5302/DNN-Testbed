from common import *
from tensorflow import keras
import argparse
#import multiprocessing as mp

def image_sender(next_socket, images, labels, data_list, lock, wait_time, arrival_rate, _stop_event):
    while True:
        # sleep before sending
        time.sleep(1/arrival_rate)

        # reading queue
        idx = np.random.randint(10000)
        data = images[idx:idx+1]
        answer = labels[idx:idx+1]
        correct = False

        # sending data
        start = time.time()
        send_data(next_socket, data)

        # make data receiving thread
        outputs = bring_data(data_list, lock, _stop_event)
        predicted = tf.argmax(outputs, 1)
        correct = (int(predicted) == int(answer[0]))

        # wait for response
        print("time took: ", time.time() - start)
        print("correct:", correct)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow')
    parser.add_argument('--set_gpu', default=True, type=str2bool, help='If you want to use GPU, set "True"')
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
    parser.add_argument('--alexnet_wait_time', default=0.500, type=float, help='waiting time for making batch')
    parser.add_argument('--googlenet_wait_time', default=0.500, type=float, help='waiting time for making batch')
    parser.add_argument('--mobilenet_wait_time', default=0.500, type=float, help='waiting time for making batch')
    parser.add_argument('--vggnet_wait_time', default=0.500, type=float, help='waiting time for making batch')
    parser.add_argument('--vggfnet_wait_time', default=0.500, type=float, help='waiting time for making batch')
    parser.add_argument('--alexnet_arrival_rate', default=30, type=int, help='arrival rate')
    parser.add_argument('--googlenet_arrival_rate', default=30, type=int, help='arrival rate')
    parser.add_argument('--mobilenet_arrival_rate', default=30, type=int, help='arrival rate')
    parser.add_argument('--vggnet_arrival_rate', default=30, type=int, help='arrival rate')
    parser.add_argument('--vggfnet_arrival_rate', default=30, type=int, help='arrival rate')
    parser.add_argument('--vram_limit', default=64, type=int, help='Next node port')
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

    '''
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
    '''

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

    alexnet_data_list = []
    '''
    googlenet_data_list = []
    mobilenet_data_list = []
    vggnet_data_list = []
    '''
    vggfnet_data_list = []

    alexnet_lock = threading.Lock()
    '''
    googlenet_lock = threading.Lock()
    mobilenet_lock = threading.Lock()
    vggnet_lock = threading.Lock()
    '''
    vggfnet_lock = threading.Lock()

    procs = []
    procs.append(threading.Thread(target=recv_data, args=(alexnet_port, alexnet_data_list, alexnet_lock, _stop_event)))
    procs.append(threading.Thread(target=image_sender, args=(alexnet_next_sock, images, labels, alexnet_data_list, alexnet_lock, args.alexnet_wait_time, args.alexnet_arrival_rate, _stop_event)))
    '''
    procs.append(threading.Thread(target=recv_data, args=(googlenet_port, googlenet_data_list, googlenet_lock, _stop_event)))
    procs.append(threading.Thread(target=image_sender, args=(googlenet_next_sock, images, labels, googlenet_data_list, googlenet_lock, args.googlenet_wait_time, args.googlenet_arrival_rate, _stop_event)))
    procs.append(threading.Thread(target=recv_data, args=(mobilenet_port, mobilenet_data_list, mobilenet_lock, _stop_event)))
    procs.append(threading.Thread(target=image_sender, args=(mobilenet_next_sock, images, labels, mobilenet_data_list, mobilenet_lock, args.mobilenet_wait_time, args.mobilenet_arrival_rate, _stop_event)))
    procs.append(threading.Thread(target=recv_data, args=(vggnet_port, vggnet_data_list, vggnet_lock, _stop_event)))
    procs.append(threading.Thread(target=image_sender, args=(vggnet_next_sock, images, labels, vggnet_data_list, vggnet_lock, args.vggnet_wait_time, args.vggnet_arrival_rate, _stop_event)))
    '''
    procs.append(threading.Thread(target=recv_data, args=(vggfnet_port, vggfnet_data_list, vggfnet_lock, _stop_event)))
    procs.append(threading.Thread(target=image_sender, args=(vggfnet_next_sock, images, labels, vggfnet_data_list, vggfnet_lock, args.vggfnet_wait_time, args.vggfnet_arrival_rate, _stop_event)))

    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()

    alexnet_next_sock.close()
    alexnet_prev_sock.close()
    '''
    googlenet_next_sock.close()
    googlenet_prev_sock.close()
    mobilenet_next_sock.close()
    mobilenet_prev_sock.close()
    vggnet_next_sock.close()
    vggnet_prev_sock.close()
    '''
    vggfnet_next_sock.close()
    vggfnet_prev_sock.close()