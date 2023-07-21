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
from dataset import VideoDataset
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


def compute_acc(pred_scores, true_scores):
    pred_scores, true_scores = np.array(pred_scores).astype(int), np.array(true_scores).astype(int)
    # TP predict 1 label 1
    TP = sum((pred_scores == 1) & (true_scores == 1))
    # TN predict 0 label 0
    TN = sum((pred_scores == 0) & (true_scores == 0))
    # FN predict 0 label 1
    FN = sum((pred_scores == 0) & (true_scores == 1))
    # FP predict 1 label 0
    FP = sum((pred_scores == 1) & (true_scores == 0))
    # print('TP, TN, FN, FP: ', TP, TN, FN, FP)
    # print(pred_scores, true_scores)

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)

    return acc


def get_dataloaders(args):
    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(VideoDataset('train', args),
                                                       batch_size=args.train_batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       worker_init_fn=worker_init_fn)

    dataloaders['test'] = torch.utils.data.DataLoader(VideoDataset('test', args),
                                                      batch_size=args.test_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn)
    return dataloaders

def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output,min=1e-7,max=1-1e-7)
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def BCELoss_ClassWeights(input, target, pos_weight, neg_weight):
    # input (n, d)
    # target (n, d)
    # class_weights (1, d)
    input = torch.clamp(input,min=1e-7,max=1-1e-7)
    loss =  pos_weight * (target * torch.log(input)) + neg_weight* ((1 - target) * torch.log(1 - input))
    return loss


def main(dataloaders, i3d, evaluator, base_logger, TB_looger, args):
    
    sheet = xlrd.open_workbook(os.path.join("./FAQA/", 'Train_knees.xls')).sheet_by_name('Sheet1')
    label_column_index = None
    for col_index in range(sheet.ncols):
        if sheet.cell_value(0, col_index) == 'label':
            label_column_index = col_index
            break
    labels = np.array(sheet.col_values(label_column_index, start_rowx=1))  # Excluye el encabezado

    num_positive = np.sum(labels == 1)
    num_negative = np.sum(labels == 0)
    weight_positive = num_negative / (num_positive + num_negative)
    weight_negative = num_positive / (num_positive + num_negative)
    print(weight_positive, weight_negative)
    
    # print configuration
    print('=' * 40)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 40)

    # TSA_Block:
    if args.TSA is True:
        TSA_module = TSA_Module(I3D_ENDPOINTS[args.TSA_pos][1], bn_layer=True).cuda()

    weights = torch.tensor([weight_negative, weight_positive]).cuda()
    #criterion_bce = nn.BCELoss(weight=weights)
    criterion_bce = nn.BCELoss()
    #criterion_bce = weighted_binary_cross_entropy()
    

    # Create Optimizer
    if args.TSA is True:
        parameters = itertools.chain(i3d.parameters(), evaluator.parameters(), TSA_module.parameters())
    else:
        parameters = itertools.chain(i3d.parameters(), evaluator.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    epoch_start, epoch_best, rho_best, train_cnt = 0, 0, 0, 0

    # Load pre-trained weights:
    if args.pt_w is not None:
        weights = torch.load(args.pt_w)
        # param of i3d
        i3d.load_state_dict(weights['i3d'])
        # param of evaluator
        evaluator.load_state_dict(weights['evaluator'])
        # param of TSA module
        if args.TSA is True:
            TSA_module.load_state_dict(weights['TSA_module'])
        # param of optimizer
        optimizer.load_state_dict(weights['optimizer'])
        # records
        epoch_start, train_cnt = weights['epoch'], weights['epoch']
        print('----- Pre-trained weight loaded from ' + args.pt_w)
    
    for epoch in range(epoch_start, args.num_epochs):
        log_and_print(base_logger, 'Epoch: %d  Current Best: %.2f at epoch %d' % (epoch, rho_best * 100, epoch_best))
        
        for split in ['train','test']:  #
            true_scores, pred_scores, keys_list = [], [], []

            # Set train/test mode of the whole model
            if split == 'train':
                i3d.train()
                evaluator.train()
                if args.TSA is True:
                    TSA_module.train()
                torch.set_grad_enabled(True)
            else:
                i3d.eval()
                evaluator.eval()
                if args.TSA is True:
                    TSA_module.eval()
                torch.set_grad_enabled(False)

            # Training / Testing
            for data in tqdm(dataloaders[split]):
                
                true_scores.extend(data['final_score'].numpy())
                videos = data['video'].cuda()
                boxes = data['boxes']  # [B,200,8]
                keys_list.extend(data['keys'])
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
                pred_scores.extend([i for i in probs.cpu().detach().numpy().reshape((-1,))])  # probs

                if split == 'train':
                    train_cnt += 1  # used for TB_looger during training

                    #loss = weighted_binary_cross_entropy(probs, data['final_score'].reshape((batch_size, -1)).float().cuda(), weights=weights)
                    loss = criterion_bce(probs, data['final_score'].reshape((batch_size, -1)).float().cuda())

                    # Save Train loss
                    TB_looger.scalar_summary(tag=split + '_BCE-loss', value=loss, step=train_cnt)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Test:
            # if split == 'test':
            pred_cls = []
            for i in range(len(pred_scores)):
                # print(keys_list[i], pred_scores[i], '\t', 1 if pred_scores[i] > 0.5 else 0, '\t GT:', true_scores[i])
                pred_cls.append(1 if pred_scores[i] > 0.5 else 0)

            acc = compute_acc(pred_cls, true_scores)
            f1score_class_0 = f1_score(pred_cls, true_scores, pos_label=0)
            f1score_class_1 = f1_score(pred_cls, true_scores, pos_label=1)
            f1score_global = f1_score(pred_cls, true_scores)

            TB_looger.scalar_summary(tag=split + '_Acc', value=acc, step=epoch)
            log_and_print(base_logger, '\t %s Acc: %.2f F-Score: %.2f F1C0: %.2F F1C1: %.2f ' % (split, acc * 100, f1score_global, f1score_class_0, f1score_class_1))

        if f1score_class_1 > rho_best:
            rho_best = f1score_class_1
            epoch_best = epoch
            log_and_print(base_logger, '-----New best found!-----')
            if args.save:
                torch.save({'epoch': epoch,
                            'i3d': i3d.state_dict(),
                            'evaluator': evaluator.state_dict(),
                            'TSA_module': TSA_module.state_dict() if args.TSA is True else None,
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
