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
import msgspec



def load_image_train(image_path, hori_flip, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0, 3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if hori_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0, 3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


class VideoDataset(Dataset):

    def __init__(self, mode, args, error):
        super(VideoDataset, self).__init__()

        self.mode = mode  # train or test
        self.args = args
        self.labels = []

        if self.mode == 'train':
            with open(os.path.join(self.args.dataset_path, 'Splits/train_keys.json'), 'r') as file_object:
                self.info = json.load(file_object)
        elif self.mode == 'test':
            with open(os.path.join(self.args.dataset_path, 'Splits/test_keys.json'), 'r') as file_object:
                self.info = json.load(file_object)
        elif self.mode == 'val':
            with open(os.path.join(self.args.dataset_path, 'Splits/val_keys.json'), 'r') as file_object:
                self.info = json.load(file_object)

        with open(os.path.join(self.args.dataset_path, 'Labels/'+error), 'r') as file_object:
                self.error = json.load(file_object)

        with open(os.path.join(self.args.dataset_path, 'ohp_boxes_32.json'), 'rb') as f:
            data = f.read()
        
        self.boxes = msgspec.json.decode(data)

        self.ann = {}
        
        for video in self.info:
            #print(video)
            if video in self.boxes:
                if len(self.error[video]) > 0: 
                    self.ann[video] = 1 
                else: 
                    self.ann[video] = 0
                self.labels.append(self.ann[video])
            
        # keys
        self.keys = list(self.ann.keys())
    
    def extract_part(self, values):
        #print(len(values))
        total_images = len(values)
        step = max(total_images // num_frames, 1)
        new_values = values[::step]

        if len(new_values) < num_frames:
            new_values.append(values[-1])
        return new_values
    
    def get_path_to_files(self, key):
        image_list = sorted((glob.glob(os.path.join(os.path.join(self.args.dataset_path, 'Images_32'), key, '*.jpg'))))
        box = self.boxes[key]
        image_list = self.extract_part(image_list)
        box =self.extract_part(box)
        return image_list, box

    def temporal_shift(self, values,shift_amount=10):
        return values[shift_amount:] + values[:shift_amount]

    def get_imgs(self, key):
        transform = transforms.Compose([transforms.CenterCrop(H), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])])
        shift = np.random.randint(0, 6)
        image_list, box = self.get_path_to_files(key)

        box = self.temporal_shift(box,shift)
        image_list = self.temporal_shift(image_list,shift)

        sample_range = np.arange(0, num_frames)  # [0, 102]
        # Padding frames
        
        tmp_a, tmp_b = box[0][6], box[0][7]
        box[0][6], box[0][7] = box[0][4], box[0][5]
        box[0][4], box[0][5] = tmp_a, tmp_b
        box_h = box

        box = np.array(box_h)  # (109, 8)
        images = torch.zeros(num_frames, C, H, W)
        for j, i in enumerate(sample_range):
            if self.mode == 'train':
                image = load_image(image_list[i], transform=transform)
            if self.mode == 'val':
                image = load_image(image_list[i], transform=transform)
            images[j] = image

        
        boxes = torch.tensor(box) 
        return images, boxes

    def __getitem__(self, ix):
        key = self.keys[ix]
        data = {}

        data['keys'] = key
        data['video'], data['boxes'] = self.get_imgs(key)
        data['final_score'] = 1 if self.ann[key] > 0.5 else 0
        return data

    def __len__(self):
        sample_pool = len(self.keys)
        return sample_pool

    def getlabels(self):
        return self.labels