import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
from scipy import stats
import pandas as pd
import json
from scipy.io import loadmat
import copy
import xlrd

from opts import *
from image_augmentations import *

from scipy.spatial.distance import euclidean
import json
import numpy as np
from fastdtw import fastdtw
import os
import cv2
import matplotlib.pyplot as plt
from  IPython.display import clear_output
import copy

class VideoDatasetSSL(Dataset):

    def __init__(self, input_path, total = 1000):
        super(VideoDatasetSSL, self).__init__()
        self.input_path = input_path
        with open(os.path.join(self.input_path, 'ohp_ssl_boxes.json'), 'r') as file_object:
            self.boxes = json.load(file_object)
        self.videos = []
        self.total = total
        self.get_list()

    def get_list(self):
        path = os.path.join(self.input_path,"Images")
        videos = os.listdir(path)
        videos.sort()
        for video_name in videos[:self.total]:
            self.videos.append(os.path.join(path,video_name))
    """
    def add_agumentations(self, frame):
        frame_np = np.array(frame)
        frame = Image.fromarray(frame_np)
        frame = hori_flip(frame)
        #frame = masking(frame, type='random', mask_amt=0.5)
        frame = masking_checker_ol(frame)
        frame = masking_checker_nool(frame)
        return frame
    
    def get_aug_video(self, frames):
        video = []
        for i in range(len(frames)):
            new_frame = self.add_agumentations(copy.deepcopy(frames[i]))
            video.append(new_frame)
        return video
    
    
    def load_images(self,path):
        files = os.listdir(path)
        files.sort(key=lambda x: int(x.split('.')[0]))
        frames = [cv2.imread(path+"/"+name, cv2.COLOR_BGR2RGB) for name in files]
        return frames
    

    def add_agumentations(self, frame):
        frame_np = np.array(frame)
        frame_pil = Image.fromarray(frame_np)
        frame_pil = hori_flip(frame_pil)
        frame_pil = masking(frame_pil, type='random', mask_amt=0.5)
        frame_pil = masking_checker_ol(frame_pil)
        frame_pil = masking_checker_nool(frame_pil)
        frame_np = np.array(frame_pil)
        frame = torch.tensor(frame_np)
        return frame
    """
    def add_agumentations(self, path_to_image, transform=None):
        image = Image.open(path_to_image)
        size = input_resize
        interpolator_idx = random.randint(0, 3)
        interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
        interpolator = interpolators[interpolator_idx]
        image = image.resize(size, interpolator)
        image = hori_flip(image)
        image = masking(image, type='random', mask_amt=0.5)
        image = masking_checker_ol(image)
        image = masking_checker_nool(image)
        if transform is not None:
            image = transform(image).unsqueeze(0)
        return image

    def load_video(self,path, add_augmentations = False):
        transform = transforms.Compose([transforms.CenterCrop(H), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])])
        files = os.listdir(path)
        files.sort(key=lambda x: int(x.split('.')[0]))
        images = torch.zeros(num_frames, C, H, W)
        for i, name in enumerate(files):
            path_to_image = path+"/"+name
            
            if add_augmentations:
                path_to_image = os.path.join(path, name)
                image_pil = Image.open(path_to_image)
                image_transformed = self.add_agumentations(path_to_image,transform)
                images[i] = image_transformed
            else:
                images[i] = self.load_single_image(path_to_image,transform)
            
            #images[i] = self.load_single_image(path_to_image, transform)
        return images

    def load_single_image(self, path_to_image, transform=None):
        image = Image.open(path_to_image)
        size = input_resize
        interpolator_idx = random.randint(0, 3)
        interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
        interpolator = interpolators[interpolator_idx]
        image = image.resize(size, interpolator)
        if transform is not None:
            image = transform(image).unsqueeze(0)
        return image
    
    def load_boxes(self, idx):
        video_path = self.videos[idx]
        name = video_path.split('/')[-1]
        b1 = self.boxes[name+"_first_half"]
        b2 = self.boxes[name+"_second_half"]
        return b1,b2,b1
    
    def adjust_points_for_horizontal_flip(self,points, frame_width):
        points_2d = np.reshape(points, (-1, 2))
        points_2d[:, 0] = frame_width - points_2d[:, 0]
        adjusted_points = points_2d.flatten()

        return adjusted_points
    
    def plot_sync_videos(self, idx):
        f1, f2, f3 = self.__getitem__(idx)
        b1, b2, b3 = self.load_boxes(idx)
        h,w,_ = f1[0].shape
        
        for i in range(len(f1)):  
            puntos1 = b1[i]
            puntos2 = b2[i]
            puntos3 = b3[i]
            if puntos1 != None: puntos3 = self.adjust_points_for_horizontal_flip(puntos3,w)
            f3_np = np.asarray(f3[i])

            if puntos1 != None:
                cv2.rectangle(f1[i],(int(puntos1[0]), int(puntos1[1])), (int(puntos1[4]), int(puntos1[5])), (0,255,0), 2)
            if puntos2 != None:
                cv2.rectangle(f2[i],(int(puntos2[0]), int(puntos2[1])), (int(puntos2[4]), int(puntos2[5])), (0,255,0), 2)
            if puntos1 != None:
                cv2.rectangle(f3_np,(int(puntos3[0]), int(puntos3[1])), (int(puntos3[4]), int(puntos3[5])), (0,255,0), 2)
            frame = np.hstack((f1[i], f2[i], f3_np))
            clear_output(wait=True)
            plt.imshow(frame)
            plt.show()

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        path_fh = os.path.join(video_path, "first_half")
        path_sh = os.path.join(video_path, "second_half")
        #f1 = self.load_images(path_fh)
        #f2 = self.load_images(path_sh)
        #f3 = self.get_aug_video(f1)
        f1 = self.load_video(path_fh)
        f2 = self.load_video(path_sh)
        f3 = self.load_video(path_fh, add_augmentations=True)
        b1, b2, b3 = self.load_boxes(idx)
        boxes1 = torch.tensor(b1)  # [200,8]
        boxes2 = torch.tensor(b2)
        boxes3 = torch.tensor(b3)
        return f1, f3, f2, boxes1, boxes3, boxes2 
    
    def __len__(self):
        sample_pool = len(self.videos)
        return sample_pool