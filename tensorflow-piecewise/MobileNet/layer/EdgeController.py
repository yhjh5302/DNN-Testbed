from common import *
from tensorflow import keras
import argparse
#import multiprocessing as mp


def get_scheduling_decision(model_name, next_sock, images, labels, label_list, label_lock, time_list, time_lock, _stop_event, num_model=1):
    schedule = [(partition_id, execution_order)]
    return schedule

def image_sender(model_name, next_sock, images, labels, label_list, label_lock, time_list, time_lock, _stop_event, num_model=1):
    for _ in range(100):
        # reading queue
        batch_size = 1
        idx = np.random.randint(10000-batch_size)
        data = images[idx:idx+batch_size]
        # req_id = (num_model * i) + model_idx[model_name]
        # data = (req_id, 0, 2, data)
        
        with label_lock:
            label_list.append(labels.flatten()[idx:idx+batch_size])
        # sending data
        send_input(next_sock, data, _stop_event)
        with time_lock:
            time_list.append(time.time())
    #_stop_event.set()

def image_recver(model_name, conn, label_list, label_lock, time_list, time_lock, _stop_event):
    init = time.time()
    while _stop_event.is_set() == False:
        # make data receiving thread
        outputs = recv_output(conn, _stop_event)
        predicted = tf.argmax(outputs, 1)
        with label_lock:
            answer = label_list.pop(0)
        correct = np.sum(predicted == answer)
        curr = time.time()

        with time_lock:
            start = time_list.pop(0)
        # wait for response
        print(model_name, "\t", curr - start, "\t", curr - init)

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

    # connection setup
    sock_list = []
    conn_list = []
    for i in range(len(addr_list)):
        addr, port = addr_list[i], port_list[i]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(600)
        sock.connect((addr, port))
        print('Node is ready, Connected by', addr+":", port)
        sock_list.append(sock)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((addr, port))
        sock.listen()
        conn, addr = sock.accept()
        print('AlexNet prev node is ready, Connected by', addr)
        conn_list.append(conn)

    input('Enter any key...')

    _stop_event = threading.Event()
    procs = []

    for i in range(len(addr_list)):
        sock, conn = sock_list[i], conn_list[i]
        label_list = []
        time_list = []
        label_lock = threading.Lock()
        time_lock = threading.Lock()
        procs.append(threading.Thread(target=image_sender, args=("MobileNet", sock, images, labels, label_list, label_lock, time_list, time_lock, _stop_event)))
        procs.append(threading.Thread(target=image_recver, args=("MobileNet", conn, label_list, label_lock, time_list, time_lock, _stop_event)))

    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()

    if args.block == False:
        next_sock.close()
        prev_sock.close()

    vid.release()
    cv2.destroyAllWindows()