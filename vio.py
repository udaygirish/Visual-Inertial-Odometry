
from queue import Queue
from threading import Thread

from config import ConfigEuRoC
from image import ImageProcessor
from msckf import MSCKF
from utils import *



class VIO(object):
    def __init__(self, config, img_queue, imu_queue, viewer=None, result_path = None, save_pred_txt = True):
        self.config = config
        self.viewer = viewer

        self.img_queue = img_queue
        self.imu_queue = imu_queue
        self.feature_queue = Queue()

        self.image_processor = ImageProcessor(config)
        self.msckf = MSCKF(config)
        self.result_path = result_path # Adding result path to save the features
        self.save_pred_txt = save_pred_txt  # Save the Predicted Pose in a txt file 
        # File format to save prediction - timestamp tx ty tz qx qy qz qw

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.vio_thread = Thread(target=self.process_feature)
        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()

        # Add # timestamp tx ty tz qx qy qz qw to the result file
        with open(self.result_path, 'w') as f:
            f.write('# timestamp tx ty tz qx qy qz qw\n')

    def process_img(self):
        while True:
            img_msg = self.img_queue.get()
            if img_msg is None:
                self.feature_queue.put(None)
                return
            # print('img_msg', img_msg.timestamp)

            if self.viewer is not None:
                self.viewer.update_image(img_msg.cam0_image)

            feature_msg = self.image_processor.stareo_callback(img_msg)

            if feature_msg is not None:
                self.feature_queue.put(feature_msg)

    def process_imu(self):
        while True:
            imu_msg = self.imu_queue.get()
            if imu_msg is None:
                return
            # print('imu_msg', imu_msg.timestamp)

            self.image_processor.imu_callback(imu_msg)
            self.msckf.imu_callback(imu_msg)

    def process_feature(self):
        while True:
            feature_msg = self.feature_queue.get()
            if feature_msg is None:
                return
            print('feature_msg', feature_msg.timestamp)
            result = self.msckf.feature_callback(feature_msg)

            if result is not None and self.viewer is not None:
                self.viewer.update_pose(result.cam0_pose)
            
            if result is not None and self.save_pred_txt:
                self.save_features(result)

    def save_features(self, result):
        timestamp = result.timestamp
        T = result.pose.t
        R = result.pose.R
        q = to_quaternion(R)
        with open(self.result_path, 'a') as f:
            f.write('%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' % (
                timestamp, T[0], T[1], T[2], q[0], q[1], q[2], q[3]))

        


if __name__ == '__main__':
    import time
    import argparse

    from dataset import EuRoCDataset, DataPublisher
    from viewer import Viewer

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='path/to/your/EuRoC_MAV_dataset/MH_01_easy', 
        help='Path of EuRoC MAV dataset.')
    parser.add_argument('--view', action='store_true', help='Show trajectory.')
    parser.add_argument('--result_path', type=str, default='results/stamped_traj_estimate.txt', help='Path to save the predicted pose')
    parser.add_argument('--save_pred', action='store_true', help='Save the predicted pose in a txt file')
    parser.add_argument('--save_video', action='store_true', help='Save the video of the trajectory')
    parser.add_argument('--video_path', type=str, default='output_video.mp4', help='Path to save the video')
    args = parser.parse_args()

    if args.view:
        viewer = Viewer(save_video=args.save_video, video_path=args.video_path)
    else:
        viewer = None

    dataset = EuRoCDataset(args.path)
    dataset.set_starttime(offset=40.)   # start from static state


    img_queue = Queue()
    imu_queue = Queue()
    # gt_queue = Queue()

    config = ConfigEuRoC()
    msckf_vio = VIO(config, img_queue, imu_queue, viewer=viewer, result_path=args.result_path, save_pred_txt=args.save_pred)


    duration = float('inf')
    ratio = 0.4  # make it smaller if image processing and MSCKF computation is slow
    imu_publisher = DataPublisher(
        dataset.imu, imu_queue, duration, ratio)
    img_publisher = DataPublisher(
        dataset.stereo, img_queue, duration, ratio)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)