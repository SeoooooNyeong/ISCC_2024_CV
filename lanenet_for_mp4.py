#!/usr/bin/python3
#-*- encoding: utf-8 -*-
import os.path as ops
import numpy as np
import torch
import cv2
import time
import math
import os
import matplotlib.pylab as plt
import sys
from tqdm import tqdm
import imageio
from dataset.dataset_utils import TUSIMPLE
from Lanenet.model2 import Lanenet
from utils.evaluation import gray_to_rgb_emb, process_instance_embedding
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
# from driving_es import *
from slidewindow import *
from std_msgs.msg import Int32
from slidewindow import *
from lanenet_util import *
import argparse

class VideoProcessor:
    def __init__(self, video_path):
        rospy.init_node("video_processor_node")
        self.pub = rospy.Publisher("/camera3", Image, queue_size=1)
        self.bridge = CvBridge()
        self.video_capture = cv2.VideoCapture(video_path)
        self.paused = False  # 초기에는 재생 상태로 설정
        self.current_frame = None


        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--frame-dir", type=str, default="./TUSIMPLE/test_clips/1494452927854312215")
        self.parser.add_argument("--gif-dir", type=str, default="/home/foscar/test_ws/src/LaneNet-PyTorch/TUSIMPLE/gif_output")
        self.parser.add_argument("--device", type=str, default='cpu')
        self.args = self.parser.parse_args()


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cuda'

        self.slidewindow = SlideWindow()
        self.x_location = 320
        self.last_x_location = 320
        self.is_detected = True
        self.current_lane = "LEFT"

        # Load the Model
        self.model_path = './TUSIMPLE/Lanenet_output/lanenet_epoch_300_batch_8.model' #내동영상으로 바꿔준다
        self.LaneNet_model = Lanenet(2, 4)
        self.LaneNet_model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
        self.LaneNet_model.to(self.device)

        self.current_frame= np.empty(shape=[0])

        self.show_img=None

        if not self.video_capture.isOpened():
            rospy.logerr("Error opening video stream or file")
            return

        rate = rospy.Rate(30)  # 30 Hz
        while not rospy.is_shutdown() and self.video_capture.isOpened():
            if not self.paused:
                self.process_video()
            else:
                self.show_current_frame()
            rate.sleep()

    def inference(self, gt_img_org):
        # BGR 순서
        org_shape = gt_img_org.shape
        gt_image = cv2.resize(gt_img_org, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)
        gt_image = gt_image / 127.5 - 1.0
        gt_image = torch.tensor(gt_image, dtype=torch.float)
        gt_image = np.transpose(gt_image, (2, 0, 1))
        gt_image = gt_image.to(self.device)
        # lane segmentation 
        binary_final_logits, instance_embedding = self.LaneNet_model(gt_image.unsqueeze(0))
        binary_final_logits, instance_embedding = binary_final_logits.to('cpu'), instance_embedding.to('cpu') 
        binary_img = torch.argmax(binary_final_logits, dim=1).squeeze().numpy()
        binary_img[0:65,:] = 0 #(0~85행)을 무시 - 불필요한 영역 제거      
        self.binary_img=binary_img.astype(np.uint8)
        self.binary_img[self.binary_img>0]=255
        # 차선 클러스터링, 색상 지정
        rbg_emb, cluster_result = process_instance_embedding(instance_embedding, binary_img,distance=1.5, lane_num=2)
        rbg_emb = cv2.resize(rbg_emb, dsize=(org_shape[1], org_shape[0]), interpolation=cv2.INTER_LINEAR)
        a = 0.1
        frame = a * gt_img_org[..., ::-1] / 255 + rbg_emb * (1 - a)
        frame = np.rint(frame * 255)
        frame = frame.astype(np.uint8)


        return frame


    def process_video(self):
        ret, frame = self.video_capture.read()

        if not ret:
            rospy.logerr("Error reading frame")
            return


        self.current_frame = frame
        self.current_frame= cv2.resize(self.current_frame,dsize=(640,480))
        self.show_img = self.inference(self.current_frame)
        self.img_frame = self.show_img.copy() # img_frame변수에 카메라 이미지를 받아옵니다.   
        height,width,channel = self.img_frame.shape
        img_roi = self.img_frame[280:,0:]   # y좌표 0~320 사이에는 차선과 관련없는 이미지들이 존재하기에 노이즈를 줄이기 위하여 roi설정을 해주었습니다.
        self.img_filtered = color_filter(img_roi)   #roi가 설정된 이미지를 color_filtering 하여 흰색 픽셀만을 추출해냅니다. 
        self.img_warped = bird_eye_view(self.img_filtered,width,height) # 앞서 구현한 bird-eye-view 함수를 이용하여 시점변환해줍니다. 
    
    
        _, L, _ = cv2.split(cv2.cvtColor(self.img_warped, cv2.COLOR_BGR2HLS))
        _, img_binary = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY) #color_filtering 된 이미지를 한번 더 이진화 하여 차선 검출의 신뢰도를 높였습니다. 

        img_masked = region_of_interest(img_binary) #이진화까지 마친 이미지에 roi를 다시 설정하여줍니다.
        self.out_img, x_location, current_lane= self.slidewindow.slidewindow(img_masked, self.is_detected)
        self.img_warped=cv2.resize(self.img_warped,dsize=(640,480))   
        self.img_blended = cv2.addWeighted(self.out_img, 1, self.img_warped, 0.6, 0) # sliding window결과를 시각화하기 위하여 out_img와 시점변환된이미지를 merging 하였습니다. 
        # out_img, x_location, current_lane= self.slidewindow.slidewindow(self.img_blended, self.is_detected)
        self.display_frames()



    def show_current_frame(self):
        self.display_frames()
        while self.paused:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('w'):
                self.paused = False  # 'w' 키를 누르면 재생 상태로 전환
            elif key == ord('q'):
                self.paused = True  # 'q' 키를 누르면 계속 일시정지

    def display_frames(self):
        cv2.imshow("original_image",self.current_frame)
        cv2.imshow("lane_image", self.show_img)
        cv2.imshow("binary_image",self.binary_img)
        cv2.imshow("img_blended",self.img_blended)
        cv2.imshow("img_wraped",self.img_warped)
        cv2.imshow("img_filtered",self.img_filtered)

        # cv2.imshow('slidewinoow',self.out_img) 
        # cv2.imshow("CAM View", self.img_blended)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.paused = True  # 'q' 키를 누르면 일시정지 상태로 전환



if __name__ == "__main__":
    video_path = "/home/donghyun/test5.mp4"  # 동영상 파일 경로를 지정하세요
    VideoProcessor(video_path)
    cv2.destroyAllWindows()





#image = np.empty(shape=[0])


