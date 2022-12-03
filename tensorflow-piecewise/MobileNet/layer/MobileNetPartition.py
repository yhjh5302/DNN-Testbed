from common import *
from MobileNetModel import MobileNet

def processing(inputs, model):
    outputs = model(inputs)
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow')
    parser.add_argument('--set_gpu', default=True, type=str2bool, help='If you want to use GPU, set "True"')
    parser.add_argument('--device_index', default=1, type=int, help='device index for device')
    parser.add_argument('--device_addr_list', default=['192.168.1.13', '192.168.1.4'], nargs='+', type=str, help='address list of kubernetes cluster')
    parser.add_argument('--prev_addr', default='localhost', type=str, help='Previous node address')
    parser.add_argument('--prev_port', default=30021, type=int, help='Previous node port')
    parser.add_argument('--next_addr', default='localhost', type=str, help='Next node address')
    parser.add_argument('--next_port', default=30020, type=int, help='Next node port')
    parser.add_argument('--vram_limit', default=1024, type=int, help='Vram limitation')
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
    model = MobileNet(name='MobileNet')
    model.conv1.load_weights('./MobileNet_conv_1_weights')
    model.separable_conv2.load_weights('./MobileNet_separable_conv2_weights')
    model.separable_conv3.load_weights('./MobileNet_separable_conv3_weights')
    model.separable_conv4.load_weights('./MobileNet_separable_conv4_weights')
    model.separable_conv5.load_weights('./MobileNet_separable_conv5_weights')
    model.separable_conv6.load_weights('./MobileNet_separable_conv6_weights')
    model.separable_conv7.load_weights('./MobileNet_separable_conv7_weights')
    model.separable_conv8.load_weights('./MobileNet_separable_conv8_weights')
    model.separable_conv9.load_weights('./MobileNet_separable_conv9_weights')
    model.separable_conv10.load_weights('./MobileNet_separable_conv10_weights')
    model.separable_conv11.load_weights('./MobileNet_separable_conv11_weights')
    model.separable_conv12.load_weights('./MobileNet_separable_conv12_weights')
    model.separable_conv13.load_weights('./MobileNet_separable_conv13_weights')
    model.separable_conv14.load_weights('./MobileNet_separable_conv14_weights')
    model.fully_connected.load_weights('./MobileNet_fully_connected_weights')

    # for cuDNN loading
    _ = model(np.zeros((1,32,32,3)))

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
    recv_lock = threading.Lock()
    send_data_list = []
    send_lock = threading.Lock()
    recv_time_list = []
    recv_time_lock = threading.Lock()
    _stop_event = threading.Event()
    threading.Thread(target=recv_data, args=(p, recv_data_list, recv_time_list, recv_lock, recv_time_lock, _stop_event)).start()
    threading.Thread(target=send_data, args=(next_sock, send_data_list, send_lock, _stop_event)).start()

    while True:
        inputs = bring_data(recv_data_list, recv_lock, _stop_event)
        outputs = processing(inputs, model)
        with send_lock:
            send_data_list.append(outputs)
        with recv_time_lock:
            print("processing time", time.time() - recv_time_list.pop(0))

    prev_sock.close()
    next_sock.close()