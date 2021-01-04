#!/usr/bin/env python3.5
# coding: utf-8
import os
import sys

CV2_ROS = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if CV2_ROS in sys.path:
    sys.path.remove(CV2_ROS)

import cv2
sys.path.append(CV2_ROS)
from cv_bridge import CvBridge, CvBridgeError

import argparse
import numpy as np
import threading
import matplotlib.pyplot as plt

import torch
import rospy
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from models import *
from utils.utils import *
from utils.datasets import *
from PIL import Image as ImagePIL

from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel
from geometry_msgs.msg import PoseStamped


class Realsense:
    """
    Class to use a Relsense camera for Hand detection
    """
    def __init__(self):
        """
        Constructor
        """
        rospy.init_node(
            "ros_yolo",
            anonymous=True,
            disable_signals=False)

        self.bridge = CvBridge()
        self.rgb_lock = threading.Lock()
        self.depth_lock = threading.Lock()

        self.cameraModel = PinholeCameraModel()
        self.cameraInfo = CameraInfo()

        self.sub_color = rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self.color_callback,
            queue_size=1)

        self.sub_depth = rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw",
            Image,
            self.depth_callback,
            queue_size=1)
        self.sub_camera_info = rospy.Subscriber(
            "/camera/depth/camera_info",
            CameraInfo,
            self.camera_info_callback,
            queue_size=1)

        self.target_pub = rospy.Publisher(
            "/hand_detection/target_camera",
            PoseStamped)

        self.color_img = np.zeros((480, 640, 3), np.uint8)
        self.depth_img = np.zeros((480, 640), np.uint8)
        self.img_width = 640
        self.img_height = 480
        self.hand_counter = 0
        self.not_hand_counter = 0
        self.hand_3D_pos = np.empty((0,3), int)

    def _random_centered_pixels(self, box_x, box_y, box_w,
                                box_h, radius, samples):
        """
        Returns an array of pixel positions in 2D
        """
        pixels = []
        center_x = int(box_w / 2)
        center_y = int(box_h / 2)
        x = box_x + center_x - radius
        y = box_y + center_y - radius
        samples_limit = radius * radius

        if samples < samples_limit:
            random_positions = random.sample(range(0, samples_limit), samples)
            for sample in random_positions:
                pixel_x = x + (sample % radius)
                if pixel_x > (self.img_width / 2 - 1):
                    pixel_x = self.img_width / 2 - 1
                pixel_y = y + int(sample / radius)
                if pixel_y > (self.img_height / 2 - 1):
                    pixel_y = self.img_height / 2 - 1
                pixels.append([pixel_x, pixel_y])
        else:
            print("The samples number should be minor"+
                  " than radius*radius")
        return pixels

    def _load_model(self, custom=True):
        if custom:
            model_def_path = "config/yolov3-custom-hand-arm.cfg"
            weights_path = "checkpoints-arm-hand/yolov3_ckpt_54.pth"
            class_path = "data/custom/classes.names"
        else:
            model_def_path = "config/yolov3.cfg"
            weights_path = "weights/yolov3.weights"
            class_path = "data/coco.names"

        self.img_size = 416

        # Object confidence threshold
        self.conf_thres = 0.95
        # IOU thresshold for non-maximum suppression
        self.nms_thres = 0.05

        if torch.cuda.is_available():
            print("---------GPU AVAILABLE---------")
            self.Tensor = torch.cuda.FloatTensor
        else:
            print("---------CPU AVAILABLE---------")
            self.Tensor = torch.FloatTensor

        # Create the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet(model_def_path, img_size=416).to(device)

        # Load the custom weights
        if custom:
            self.model.load_state_dict(torch.load(weights_path))
        else:
            self.model.load_darknet_weights(weights_path)

        self.model.eval()

        self.classes = load_classes(class_path)

        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 10000)]
        self.bbox_colors = random.sample(colors, len(self.classes))

    def _infer(self):
        with self.rgb_lock:
            frame = self.color_img.copy()
            # frame = self.color_img
        # Display only purpose
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImagePIL.fromarray(frame)
        ratio = min(self.img_size / img.size[0],
                    self.img_size / img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([
            transforms.Resize((imh, imw)),
            transforms.Pad((max(int((imh - imw) / 2), 0),
                            max(int((imw - imh) / 2), 0), max(int((imh - imw) / 2),0),
                            max(int((imw - imh) / 2), 0)), (128, 128, 128)),
            transforms.ToTensor(),
            ])

        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(self.Tensor))

        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(
                detections,
                self.conf_thres,
                self.nms_thres)

        if detections[0] is not None:

            unit_scaling = 0.001
            center_x = self.cameraModel.cx()
            center_y = self.cameraModel.cy()
            constant_x = unit_scaling / self.cameraModel.fx()
            constant_y = unit_scaling / self.cameraModel.fy()

            # Rescale boxes to original image
            detections = rescale_boxes(
                detections[0],
                self.img_size,
                frame.shape[:2])

            unique_labels = detections[:, -1].cpu().unique()

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                box_w = x2 - x1
                box_h = y2 - y1
                w_proportion = box_w / img.size[0]
                h_proportion = box_h / img.size[1]
                if ((self.classes[int(cls_pred)] == 'Human hand' or
                     self.classes[int(cls_pred)] == 'Human arm') and
                        w_proportion >= 0.1 and w_proportion <= 0.90 and
                        h_proportion >= 0.1 and h_proportion <= 1.0 and
                        box_w > 10 and box_h > 10):

                    color = self.bbox_colors[
                        int(np.where(unique_labels == int(cls_pred))[0])]

                    frame = cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (color[0] * 255, color[1] * 255, color[2] * 255),
                        2)

                    cv2.putText(
                        frame,
                        self.classes[int(cls_pred)],
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA)

                    radius = 5
                    samples = 10
                    pixels = self._random_centered_pixels(
                             float(x1),
                             float(y1),
                             box_w,
                             box_h,
                             radius,
                             samples)
                    depthPoint = []
                    arrayPoints = np.empty((0, 3), int)

                    for i in range(0, samples - 1):
                        x = int(pixels[i][0])
                        y = int(pixels[i][1])
                        depthPoint = np.array([
                            (x - center_x / 2) *
                            self.depth_img[y, x] * constant_x,
                            (y - center_y / 2) *
                            self.depth_img[y, x] * constant_y,
                            self.depth_img[y, x] * unit_scaling])
                        if depthPoint.sum() > 0:
                            arrayPoints = np.append(
                                              arrayPoints,
                                              [depthPoint],
                                              axis=0)

                    if len(arrayPoints) > 0:
                        mean_x = np.mean(arrayPoints[:, 0], axis=0)
                        mean_y = np.mean(arrayPoints[:, 1], axis=0)
                        mean_z = np.mean(arrayPoints[:, 2], axis=0)

                        pos_3D = np.array([mean_x, mean_y, mean_z])
                        if self.hand_counter == 0:
                            self.hand_3D_pos = pos_3D
                            self.hand_counter = 1
                        elif mean_z > 0.25 and mean_z < 1.20:
                            dist = np.linalg.norm(self.hand_3D_pos - pos_3D)
                            if ((dist < 0.10) or
                               (pos_3D[2] < self.hand_3D_pos[2])):
                                self.hand_3D_pos = pos_3D
                                self.hand_counter = self.hand_counter + 1

                                if self.hand_counter > 2:
                                    # if self.hand_counter == 3:
                                    #    cv2.imwrite('hand.jpg', frame)

                                    header = std_msgs.msg.Header()
                                    header.stamp = rospy.Time.now()
                                    header.frame_id = 'RealSense_optical_frame'
                                    # publish
                                    p = PoseStamped()
                                    p.header.frame_id = 'target_camera'
                                    p.header.stamp = rospy.Time.now()

                                    p.pose.position.x = mean_x
                                    p.pose.position.y = mean_y
                                    p.pose.position.z = mean_z
                                    self.target_pub.publish(p)
                                    break
                            else:
                                self.not_hand_counter = self.not_hand_counter+1

            if self.not_hand_counter > 20:
                self.hand_counter = 0
                self.not_hand_counter = 0

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    def color_callback(self, data):
        """
        Callback function for color image subscriber
        """
        try:
            with self.rgb_lock:
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                self.color_img = cv_image
        except CvBridgeError as e:
            print(e)

    def depth_callback(self, data):
        """
        Callback function for depth image subscriber
        """
        try:
            with self.depth_lock:
                cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
                self.depth_img = np.array(cv_image, dtype=np.float32)
        except CvBridgeError as e:
            print(e)

    def camera_info_callback(self, data):
        """
        Callback function for camera info subscriber
        """
        try:
            self.cameraModel.fromCameraInfo(data)
            self.cameraInfo = data
        except Error as e:
            print(e)


if __name__ == "__main__":
    node = Realsense()
    node._load_model(custom=True)

    try:
        rate = rospy.Rate(30.0)
        while not rospy.is_shutdown():
            node._infer()
            rate.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
