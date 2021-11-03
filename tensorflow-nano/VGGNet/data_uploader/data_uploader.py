from common import *
from tensorflow import keras
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow')
    parser.add_argument('--set_gpu', default=False, type=str2bool, help='If you want to use GPU, set "True"')
    parser.add_argument('--prev_addr', default='10.96.0.200', type=str, help='Previous node address')
    parser.add_argument('--prev_port', default=30030, type=int, help='Previous node port')
    parser.add_argument('--next_addr', default='10.96.0.231', type=str, help='Next node address')
    parser.add_argument('--next_port', default=30031, type=int, help='Next node port')
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

    next_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    next_sock.settimeout(600)
    next_sock.connect((args.next_addr, args.next_port))
    print('Next node is ready, Connected by', args.next_addr)

    prev_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    prev_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    prev_sock.bind((args.prev_addr, args.prev_port))
    prev_sock.listen()
    p, addr = prev_sock.accept()
    print('Previous node is ready, Connected by', addr)

    # make data receiving thread
    data_list = []
    lock = threading.Lock()
    _stop_event = threading.Event()
    threading.Thread(target=recv_data, args=(p, data_list, lock, _stop_event)).start()

    while True:
        # input number of images
        try:
            total = int(input('Enter num of images(zero is quit): '))
        except:
            print('Please enter positive integer')
            continue
        if total == 0:
            break

        batch_size = 32
        total_images = total * batch_size
        correct = 0
        answer = np.zeros(total_images)

        # sending data
        total_start = time.time()
        for idx in range(total):
            img_idx = np.random.randint(10000-batch_size)
            send_data(next_sock, images[img_idx:img_idx+batch_size])
            answer[batch_size*idx:batch_size*(idx+1)] = labels.flatten()[img_idx:img_idx+batch_size]

        print("data sending took: {:.5f} sec".format(time.time() - total_start))

        # wait for response
        for idx in range(total):
            outputs = bring_data(data_list, lock, _stop_event)
            predicted = tf.argmax(outputs, 1)
            correct += np.sum(predicted == answer[batch_size*idx:batch_size*(idx+1)])

        print("time took: {:.5f} sec".format(time.time() - total_start))
        print("correct:", correct)
        print("total:", total_images)
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total_images))

    prev_sock.close()
    next_sock.close()