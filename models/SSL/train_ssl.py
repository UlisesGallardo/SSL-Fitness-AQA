import os
import sys

sys.path.append('../')

import torch
import torch.nn as nn

from scipy import stats
from tqdm import tqdm
import itertools

from models.i3d import InceptionI3d
from models.i3d import TSA_Module, NONLocalBlock3D
from models.evaluator import Evaluator, get_mask

from opts import *
from datasetSSL import VideoDatasetSSL
from config import get_parser
from logger import Logger

from utils import *

from thop import profile
from thop import clever_format
from sklearn.metrics import f1_score
import xlrd


def get_models(args):
    """
    Get the i3d backbone and the evaluator with parameters moved to GPU.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    i3d = InceptionI3d().cuda()
    i3d.load_state_dict(torch.load(i3d_pretrained_path))

    evaluator = Evaluator(output_dim=output_dim['USDL'], model_type='USDL').cuda()

    if len(args.gpu.split(',')) > 1:
        i3d = nn.DataParallel(i3d)
        evaluator = nn.DataParallel(evaluator)
    return i3d, evaluator

def get_dataloaders(args):
    dataloaders = {}
    input_path = "./OHP_Unlabeled/"

    dataloaders['train'] = torch.utils.data.DataLoader(VideoDatasetSSL(input_path, 200),
                                                       batch_size=args.train_batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       worker_init_fn=worker_init_fn)
    return dataloaders

class DistanceRatioLoss(nn.Module):
    
    def __init__(self):
        super(DistanceRatioLoss, self).__init__()
    
    def forward(self, anchor, positive, negative):
        epsilon = 1e-8
        dist_pos = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), dim=1))
        dist_neg = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), dim=1))
        dist_pos = torch.exp(-dist_pos)
        dist_neg = torch.exp(-dist_neg)
        loss = -torch.log((dist_pos+epsilon)/(dist_pos+dist_neg+epsilon))
        loss = torch.mean(torch.clamp(loss, min = 0))
        return loss

def train_ssl_model(videos, boxes, TSA_module): #boxes [B,200,8]
    videos = videos.cuda()
    videos.transpose_(1, 2)  # N, C, T, H, W
    batch_size, C, frames, H, W = videos.shape
    clip_feats = torch.empty(batch_size, seg_number, feature_dim).cuda()
    
    ### Forward
    if args.TSA is False:
        for i in range(seg_number):  # [4,3,200,224,224]=>[4,1024]
            clip_feats[:, i] = i3d(videos[:, :, frames_seg * i:frames_seg * i + frames_seg, :, :], args).squeeze(2)
    else:
        # ####
        # Stage 1 of I3D
        # ####
        feats_tsa = []
        for i in range(seg_number):  # [0,TSA_loc]
            #print("start",20*i, "end",20*i + 20)
            feats_tsa.append(i3d(videos[:, :, frames_seg * i:frames_seg * i + frames_seg, :, :], args, stage=1))  # [B,C,T,H,W]
        ckpt_C, ckpt_T, ckpt_S = feats_tsa[0].shape[1:4]
        
        #print("Stage 1:", ckpt_C, ckpt_T, ckpt_S)
        #print("Stage 1: Features",np.array(feats_tsa).cpu().numpy().shape)
        # ####
        # Feature enhancement stage (FLOPS counter)
        # ####
        feats_tsa = torch.cat(feats_tsa, dim=2).cuda()  # [B,C,T*10,H,W] Merge time
        mask = get_mask(x=feats_tsa, boxes=boxes, img_size=(W_img, H_img))  # Get box : [B,T,H,W]

        #print("Feature enhancement: Features",feats_tsa.shape, "Mask",mask.shape)
        feats_tsa = TSA_module(feats_tsa, mask)

        feats_tsa = feats_tsa.view(batch_size, ckpt_C, ckpt_T, seg_number, ckpt_S, ckpt_S)
        feats_tsa = feats_tsa.permute(0, 1, 2, 4, 5, 3).contiguous()  # [4,192,8,28,28,10]

        # ####
        # Stage 2 of I3D
        # ####
        #print(feats_tsa.shape)
        for i in range(seg_number):  # (TSA_loc,-1]    [4,3,200,224,224]=>[4,1024]
            res = i3d(feats_tsa[:, :, :, :, :, i], args, stage=2)
            #print(res.shape)
            #res = res.squeeze(2)
            res = torch.mean(res,-1)
            #print(res.shape)
            clip_feats[:, i] = res
            #clip_feats[:, num_segments - 1] = i3d(feats_tsa[:, :, :, :, :, num_segments - 1], args, stage=2).squeeze(2)
        del feats_tsa

    probs = evaluator(clip_feats.mean(1), args)  # [4,1]
    return probs

def main(dataloaders, i3d, evaluator, base_logger, TB_looger, args):
    # print configuration
    print('=' * 40)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 40)

    # TSA_Block:
    if args.TSA is True:
        TSA_module_ = TSA_Module(I3D_ENDPOINTS[args.TSA_pos][1], bn_layer=True).cuda()

    criterion = DistanceRatioLoss()
    
    # Create Optimizer
    if args.TSA is True:
        parameters = itertools.chain(i3d.parameters(), evaluator.parameters(), TSA_module_.parameters())
    else:
        parameters = itertools.chain(i3d.parameters(), evaluator.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    epoch_start, epoch_best, rho_best, train_cnt = 0, 0, 1, 0

    # Load pre-trained weights:
    if args.pt_w is not None:
        weights = torch.load(args.pt_w)
        # param of i3d
        i3d.load_state_dict(weights['i3d'])
        # param of evaluator
        evaluator.load_state_dict(weights['evaluator'])
        # param of TSA module
        if args.TSA is True:
            TSA_module_.load_state_dict(weights['TSA_module'])
        # param of optimizer
        optimizer.load_state_dict(weights['optimizer'])
        # records
        epoch_start, train_cnt = weights['epoch'], weights['epoch']
        print('----- Pre-trained weight loaded from ' + args.pt_w)
    
    for epoch in range(epoch_start, args.num_epochs):
        log_and_print(base_logger, 'Epoch: %d  Current Best: %.3f at epoch %d' % (epoch, rho_best * 100, epoch_best))
        train_loss = 0.0
        i3d.train()
        evaluator.train()
        if args.TSA is True:
            TSA_module_.train()
        torch.set_grad_enabled(True)
        split = 'train'
        # Training
        for anchor, positive, negative, anchor_boxes, positive_boxes, negative_boxes in tqdm(dataloaders[split]):
            train_cnt += 1  # used for TB_looger during training

            probs1 = train_ssl_model(anchor,anchor_boxes,TSA_module_)
            probs2 = train_ssl_model(positive,positive_boxes,TSA_module_)
            probs3 = train_ssl_model(negative,negative_boxes,TSA_module_)
            loss = criterion(probs1,probs2, probs3)
            # Save Train loss
            TB_looger.scalar_summary(tag=split + '_BCE-loss', value=loss, step=train_cnt)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()      
            #print(f"Iteration: {train_cnt}, Loss: {loss.item()}")
        
        train_loss /= len(dataloaders[split])
        log_and_print(base_logger, '\t %s Loss: %.3f ' % (split, train_loss))
        if train_loss < rho_best:
            rho_best = train_loss
            epoch_best = epoch
            log_and_print(base_logger, '-----New best found!-----')
            if args.save:
                torch.save({'epoch': epoch,
                            'i3d': i3d.state_dict(),
                            'evaluator': evaluator.state_dict(),
                            'TSA_module': TSA_module_.state_dict() if args.TSA is True else None,
                            'optimizer': optimizer.state_dict(),
                            'rho_best': rho_best},
                            f'{args.model_path}/best.pt')
            

if __name__ == '__main__':

    args = get_parser().parse_args()

    # Create Experiments dirs
    if not os.path.exists('./Exp'):
        os.mkdir('./Exp')
    args.model_path = './Exp/' + args.model_path
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    # Create Logger
    TB_looger = Logger(args.model_path + '/tb_log')

    init_seed(args)

    print(args.TSA)

    #base_logger = get_logger(f'{args.model_path}/train.log', args.log_info)
    base_logger = get_logger(f'{args.model_path}.log', args.log_info)
    i3d, evaluator = get_models(args)
    dataloaders = get_dataloaders(args)

    main(dataloaders, i3d, evaluator, base_logger, TB_looger, args)
