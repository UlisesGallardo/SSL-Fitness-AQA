import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import json

from opts import *


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

        self.mode = mode  
        self.args = args
        self.labels = []

        # Loading annotations
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

        self.ann = {}
        
        for video in self.info:
            if len(self.error[video]) > 0: 
                self.ann[video] = 1 
            else: 
                self.ann[video] = 0
            self.labels.append(self.ann[video])
    
        # keys
        self.keys = list(self.ann.keys())

    def extract_part(self, values):
        total_images = len(values)
        step = max(total_images // num_frames, 1)
        new_values = values[::step]

        if len(new_values) < num_frames:
            new_values.append(values[-1])
        return new_values
    
    def get_path_to_files(self, key):
        image_list = sorted((glob.glob(os.path.join(os.path.join(self.args.dataset_path, 'Images_200'), key, '*.jpg'))))
        image_list = self.extract_part(image_list)
        return image_list 


    def get_imgs(self, key):
        transform = transforms.Compose([transforms.CenterCrop(H), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])])

        image_list = self.get_path_to_files(key)
        sample_range = np.arange(0, num_frames)  

        images = []

        for j, i in enumerate(sample_range):
            #print(i)
            if self.mode == 'train':
                image = load_image_train(image_list[i], False, None)
            if self.mode == 'test' or self.mode == 'val':
                image = load_image(image_list[i], None)
            images.append(image)
            
        num_to_extract = 32
        total_images = len(images)
        step = max(total_images // num_to_extract, 1)
        images = images[::step]

        if len(images) < num_to_extract:
            images.append(images[-1])
            
        images_final = torch.zeros(num_frames, C, H, W)

        for i, image in enumerate(images):
            images_final[i] = transform(image).unsqueeze(0)

        return images_final

    def __getitem__(self, ix):
        key = self.keys[ix]
        data = {}

        data['keys'] = key
        data['video'] = self.get_imgs(key)
        data['final_score'] = self.ann[key]
        return data

    def getlabels(self):
        return self.labels

    def __len__(self):
        sample_pool = len(self.keys)
        return sample_pool
