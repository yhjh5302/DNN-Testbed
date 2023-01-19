from common import *
import cv2, logging


def run_yolov5_single(fps):
    # video data load
    vid = cv2.VideoCapture(args.data_path+args.video_name)
    # fps = vid.get(cv2.CAP_PROP_FPS)
    delay = int(600/fps)
    # roi_mask = cv2.imread(args.data_path+args.roi_name, cv2.IMREAD_UNCHANGED)
    # roi_mask = cv2.resize(roi_mask, args.resolution, interpolation=cv2.INTER_CUBIC)


    #TODO: add in Yolov5
    

    while vid.isOpened():
        _, frame = vid.read()

        if frame is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue
        frame = cv2.resize(frame, args.resolution, interpolation=cv2.INTER_CUBIC)
        detected = False

        # calculate the foreground mask
        took = time.time()

        if args.verbose:
            print("mask {:.5f} ms".format((time.time() - took) * 1000))

        if cv2.waitKey(delay) == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Piecewise Partition and Scheduling')
    parser.add_argument('--vram_limit', default=0.2, type=float, help='GPU memory limit')
    parser.add_argument('--master_addr', default='localhost', type=str, help='Master node ip address')
    parser.add_argument('--master_port', default='30000', type=str, help='Master node port')
    # parser.add_argument('--rank', default=0, type=int, help='Master node port', required=True)
    parser.add_argument('--data_path', default='./Data/', type=str, help='Image frame data path')
    parser.add_argument('--video_name', default='vdo.avi', type=str, help='Video file name')
    parser.add_argument('--roi_name', default='roi.jpg', type=str, help='RoI file name')
    parser.add_argument('--num_proc', default=2, type=int, help='Number of processes')
    parser.add_argument('--resolution', default=(854, 480), type=tuple, help='Image resolution')
    parser.add_argument('--fps', default=10, type=int, help='frames per second')
    parser.add_argument('--verbose', default=False, type=str2bool, help='If you want to print debug messages, set True')
    args = parser.parse_args()

    # gpu setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(fraction=args.vram_limit, device=device)
    print(device, torch.cuda.get_device_name(0))

    run_yolov5_single(args.fps)
