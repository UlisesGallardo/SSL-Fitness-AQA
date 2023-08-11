error = 'error_knees.json'
# i3d model pretrained on Kinetics, https://github.com/yaohungt/Gated-Spatio-Temporal-Energy-Graph
i3d_pretrained_path = './models/rgb_i3d_pretrained.pt'

# num of frames in a single video
num_frames = 32
seg_number = 1
frames_seg = num_frames//seg_number

# input data dims;
C, H, W = 3, 224, 224
# image resizing dims;
input_resize = 320, 320

# output dimension of I3D backbone
feature_dim = 1024

H_img, W_img = 480, 480

I3D_ENDPOINTS = {  # [name,channel,T,size]
    0: ['Conv3d_1a_7x7', 64, 8, 112],
    1: ['MaxPool3d_2a_3x3', 64, 8, 56],
    2: ['Conv3d_2b_1x1', 64, 8, 56],
    3: ['Conv3d_2c_3x3', 12, 8, 56],
    4: ['MaxPool3d_3a_3x3', 192, 8, 28],
    5: ['Mixed_3b', 256, 8, 28],
    6: ['Mixed_3c', 480, 8, 28],
    7: ['MaxPool3d_4a_3x3', 480, 4, 14],  # exp_25
    8: ['Mixed_4b', 512, 4, 14],  # exp_26
    9: ['Mixed_4c', 512, 4, 14],  # exp_27
    10: ['Mixed_4d', 512, 4, 14],  # exp_28
    11: ['Mixed_4e', 528, 4, 14],  # exp_29
    12: ['Mixed_4f', 832, 4, 14],  # exp_30
    13: ['MaxPool3d_5a_2x2', 832, 2, 7],
    14: ['Mixed_5b', 832, 2, 7],  # exp_31
    15: ['Mixed_5c', 1024, 2, 7],  # exp_32
    16: ['Logits'],
    17: ['Predictions'],
}
