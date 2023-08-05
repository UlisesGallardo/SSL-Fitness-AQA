import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from optsSSL import *

import numpy as np
import os
import cv2
import imgaug.augmenters as iaa

class VideoDatasetSSL(Dataset):

    def __init__(self, input_path, total = 1000):
        super(VideoDatasetSSL, self).__init__()
        self.input_path = input_path
        self.videos = []
        self.total = total
        self.get_list()

    def get_list(self):
        path = os.path.join(self.input_path,"Images")
        videos = os.listdir(path)
        videos.sort()
        for video_name in videos[:self.total]:
            self.videos.append(os.path.join(path,video_name))

    def load_video(self,path, num_to_extract = 16):
        files = os.listdir(path)
        files.sort(key=lambda x: int(x.split('.')[0]))

        total_images = len(files)
        step = max(total_images // num_to_extract, 1)
        files = files[::step]

        if len(files) < num_to_extract:
            files.append(files[-1])

        images = []
        for i, name in enumerate(files):
            path_to_image = path+"/"+name
            image = cv2.imread(path_to_image)
            im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(im_rgb)
        return images

    def temporal_shift(self, frames,shift_amount=3):
        return frames[shift_amount:] + frames[:shift_amount]
    
    def RandomHorizontalFlip(self, clip):
        random_flip_prob = random.choice([0.0, 1.0])

        if(random_flip_prob == 1.0):
            clip_flipped = np.flip(clip, axis=2)
            return clip_flipped
        return clip
        
    def apply_channel_shuffle(self,images, random_state, parents, hooks):
        num_channels = 3  
        channel_permutation = np.random.permutation(num_channels)
        return [image[:, :, channel_permutation] for image in images]
    
    def keypoint_func(self,keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def add_video_transforms(self, frames, shift_amount):
        frames = self.temporal_shift(frames, shift_amount)
        frames = self.RandomHorizontalFlip(frames)
        
        rotation_angle = np.random.randint(-10, 10)
        sigma_value  = np.random.uniform(0, 1.0) 
        saturation = np.random.randint(0,50)
        translate = np.random.randint(-10,10)
        scale_factor = 1.2

        video_augmenter = iaa.Sequential([
            #iaa.CoarseDropout((0.1, 0.15), size_percent=(0.03, 0.03)),
            iaa.Cutout(fill_mode="constant", size =0.3, cval=0, squared=False),
            iaa.WithHueAndSaturation(
                iaa.WithChannels(0, iaa.Add(saturation))
            ),
            iaa.Lambda(self.apply_channel_shuffle, self.keypoint_func), 
            iaa.Affine(rotate=(rotation_angle), scale=scale_factor),
            iaa.TranslateX(px=translate), 
            iaa.GaussianBlur(sigma=sigma_value), 
        ], random_order=False)

        video_ = video_augmenter(images=frames)
        return video_

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        path_fh = os.path.join(video_path, "f_h")
        path_sh = os.path.join(video_path, "s_h")
        f_h_anchor = self.load_video(path_fh)
        s_h_negative = self.load_video(path_sh)
        f_h_positive = self.load_video(path_fh)
        
        shift = np.random.randint(0, 25)
        f_h_anchor_aug  = self.add_video_transforms(f_h_anchor, shift)
        s_h_negative_aug = self.add_video_transforms(s_h_negative, shift)
        f_h_positive_aug = self.add_video_transforms(f_h_positive, shift)

        #f_h_anchor_aug = f_h_anchor
        #s_h_negative_aug = s_h_negative

        transform = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(H), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])])

        images_anc = torch.zeros(num_frames,C, H, W) #num_frames, C, H, W
        images_pos = torch.zeros(num_frames,C, H, W)
        images_neg = torch.zeros(num_frames,C, H, W)

        for i in range(num_frames):
            images_anc[i] = transform(f_h_anchor_aug[i])
            images_pos[i] = transform(f_h_positive_aug[i])
            images_neg[i] = transform(s_h_negative_aug[i])
    
        images_anc = torch.reshape(images_anc, (3, num_frames, H, W))
        images_pos = torch.reshape(images_pos, (3, num_frames, H, W))
        images_neg = torch.reshape(images_neg, (3, num_frames, H, W))
       
        return images_anc, images_pos, images_neg
    
    def __len__(self):
        sample_pool = len(self.videos)
        return sample_pool