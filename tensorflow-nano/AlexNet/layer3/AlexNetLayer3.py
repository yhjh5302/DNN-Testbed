from common import *
from AlexNetModel import *
import argparse

def processing(inputs, model):
    outputs = model(inputs)
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow')
    parser.add_argument('--set_gpu', default=True, type=str2bool, help='If you want to use GPU, set "True"')
    parser.add_argument('--prev_addr', default='10.96.0.203', type=str, help='Previous node address')
    parser.add_argument('--prev_port', default=30003, type=int, help='Previous node port')
    parser.add_argument('--next_addr', default='10.96.0.200', type=str, help='Next node address')
    parser.add_argument('--next_port', default=30000, type=int, help='Next node port')
    parser.add_argument('--vram_limit', default=1024, type=int, help='Next node port')
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
    model = AlexNet_layer_3(name='layer3')
    model.classifier_1.load_weights('./alexnet_classifier_1_weights')
    model.classifier_2.load_weights('./alexnet_classifier_2_weights')
    model.classifier_3.load_weights('./alexnet_classifier_3_weights')
    
    # for cuDNN loading
    model(np.zeros((1,6,6,256)))
    
    print('Pretrained model loading done!')
    
    prev_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    prev_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    prev_sock.bind((args.prev_addr, args.prev_port))
    prev_sock.listen()
    p, addr = prev_sock.accept()
    print('Previous node is ready, Connected by', addr)
    
    next_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    next_sock.settimeout(600) # 10 minutes
    next_sock.connect((args.next_addr, args.next_port))
    print('Next node is ready, Connected by', args.next_addr)
    
    # for time record
    total, took1, took2, took3 = 0, 0, 0, 0
    
    # for data multi-processing
    data_list = []
    lock = threading.Lock()
    _stop_event = threading.Event()
    threading.Thread(target=recv_data, args=(p, data_list, lock, _stop_event)).start()
    
    try:
        while True:
            start = time.time()
            inputs = bring_data(data_list, lock, _stop_event)
            took1 += time.time() - start
            outputs = processing(inputs, model)
            took2 += time.time() - start
            send_data(next_sock, outputs)
            took3 += time.time() - start
            total += 1
            if total >= args.debug:
                print("----------------------------------------")
                print("bring data time: {:.5f} sec".format(took1/total))
                print("processing time: {:.5f} sec".format((took2-took1)/total))
                print("communication time: {:.5f} sec".format((took3-took2)/total))
                print("output shape:", outputs.shape)
                total, took1, took2, took3 = 0, 0, 0, 0
    except:
        print("connection lost:", addr)

    prev_sock.close()
    next_sock.close()