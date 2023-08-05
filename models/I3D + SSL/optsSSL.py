# directory containing frames
frames_dir = './data/frames'
# directory containing labels and annotations
info_dir = './data/info'

# i3d model pretrained on Kinetics, https://github.com/yaohungt/Gated-Spatio-Temporal-Energy-Graph
i3d_pretrained_path = './models/rgb_i3d_pretrained.pt'

# num of frames in a single video
num_frames = 16

# input data dims;
C, H, W = 3, 224, 224

# image resizing dims;
input_resize = 320, 320

# output dimension of I3D backbone
feature_dim = 1024

H_img, W_img = 480, 480
