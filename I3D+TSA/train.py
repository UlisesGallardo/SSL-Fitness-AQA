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


def get_models(args):
    """
    Get the i3d backbone and the evaluator with parameters moved to GPU.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    i3d = InceptionI3d().cuda()
    i3d.load_state_dict(torch.load(i3d_pretrained_path))

    evaluator = Evaluator(output_dim=1).cuda()

    if len(args.gpu.split(',')) > 1:
        i3d = nn.DataParallel(i3d)
        evaluator = nn.DataParallel(evaluator)
    return i3d, evaluator


def compute_f1(pred_scores, true_scores):
    pred_scores, true_scores = np.array(pred_scores).astype(int), np.array(true_scores).astype(int)

    # Asegurarse de que las etiquetas sean binarias (0 o 1)
    pred_scores[pred_scores > 1] = 1
    true_scores[true_scores > 1] = 1

    # TP predict 1 label 1
    TP = sum((pred_scores == 1) & (true_scores == 1))
    # TN predict 0 label 0
    TN = sum((pred_scores == 0) & (true_scores == 0))
    # FN predict 0 label 1
    FN = sum((pred_scores == 0) & (true_scores == 1))
    # FP predict 1 label 0
    FP = sum((pred_scores == 1) & (true_scores == 0))

    # Verificar divisiones por cero y calcular precisión, sensibilidad y F1
    if (TP + FP) > 0:
        p = TP / (TP + FP)
    else:
        p = 0

    if (TP + FN) > 0:
        r = TP / (TP + FN)
    else:
        r = 0

    if (r + p) > 0:
        F1 = 2 * r * p / (r + p)
    else:
        F1 = 0

    acc = (TP + TN) / (TP + TN + FP + FN)

    return acc, F1


def get_dataloaders(args):
    dataloaders = {}
    train = VideoDataset('train', args, error)
    lables = train.getlabels()
    dataloaders['train'] = torch.utils.data.DataLoader(train,
                                                       batch_size=args.train_batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       worker_init_fn=worker_init_fn)

    dataloaders['val'] = torch.utils.data.DataLoader(VideoDataset('val', args, error),
                                                      batch_size=args.test_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn)
    return dataloaders, lables

#https://www.kaggle.com/code/parthdhameliya77/class-imbalance-weighted-binary-cross-entropy
#https://stackoverflow.com/questions/46009619/keras-weighted-binary-crossentropy
#https://gist.github.com/nasimrahaman/a5fb23f096d7b0c3880e1622938d0901
class W_BCEWithLogitsLoss(torch.nn.Module): 
    
    def __init__(self, w_p = None, w_n = None):
        super(W_BCEWithLogitsLoss, self).__init__()
        
        self.w_p = w_p
        self.w_n = w_n
        
    def forward(self, ps, labels, epsilon = 1e-7):
        
        loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(ps + epsilon))
        loss_neg = -1 * torch.mean(self.w_n * (1-labels) * torch.log((1-ps) + epsilon))
        
        loss = loss_pos + loss_neg
        
        return loss

def main(dataloaders, i3d, evaluator, base_logger, TB_looger, args, labels):
    
    labels = np.array(labels)
    num_positive = np.sum(labels == 1)
    num_negative = np.sum(labels == 0)
    weight_positive = num_negative / (num_positive + num_negative)
    weight_negative = num_positive / (num_positive + num_negative)
    #print(weight_positive, weight_negative)
    # print configuration
    log_and_print(base_logger, '=' * 40)
    for k, v in vars(args).items():
        log_and_print(base_logger, f'{k}: {v}')
    log_and_print(base_logger, '=' * 40)

    # TSA_Block:
    if args.TSA is True:
        TSA_module = TSA_Module(I3D_ENDPOINTS[args.TSA_pos][1], bn_layer=True).cuda()

    #weights = torch.tensor([weight_negative, weight_positive]).to("cuda")
    #criterion_bce = nn.BCELoss()
    #criterion_bce = W_BCEWithLogitsLoss(w_p=weight_positive, w_n=weight_negative) 
    weights = torch.tensor([2.0]).to("cuda")
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight = weights)

    # Create Optimizer
    if args.TSA is True:
        parameters = itertools.chain(i3d.parameters(), evaluator.parameters(), TSA_module.parameters())
    else:
        parameters = itertools.chain(i3d.parameters(), evaluator.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    epoch_start, epoch_best, rho_best, train_cnt = 0, 0, 0, 0

    # Load pre-trained weights:
    if args.pt_w:
        weights = torch.load(args.pt_w)
        # param of i3d
        i3d.load_state_dict(weights['i3d'])
        if args.TSA is True:
            TSA_module.load_state_dict(weights['TSA_module'])
        # param of optimizer
        optimizer.load_state_dict(weights['optimizer'])
        # records
        epoch_start, train_cnt = weights['epoch'], weights['epoch']
        print('----- Pre-trained weight loaded from ' + args.pt_w)
    
    for epoch in range(epoch_start, args.num_epochs):
        log_and_print(base_logger, 'Epoch: %d  Current Best: %.2f at epoch %d' % (epoch, rho_best * 100, epoch_best))
        
        for split in ['train','val']:  #
            true_scores, pred_scores, keys_list = [], [], []
            losses = 0.0
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
                    for i in range(seg_number):  # (TSA_loc,-1]    [4,3,32,224,224]=>[4,1024]
                        res = i3d(feats_tsa[:, :, :, :, :, i], args, stage=2)
                        res = torch.mean(res,-1)
                        clip_feats[:, i] = res
                    del feats_tsa

                probs = evaluator(clip_feats.mean(1))  # [4,1]
                pred_scores.extend([i for i in probs.cpu().detach().numpy().reshape((-1,))])  # probs

                if split == 'train':
                    train_cnt += 1  # used for TB_looger during training

                    #loss = weighted_binary_cross_entropy(probs, data['final_score'].reshape((batch_size, -1)).float().cuda(), weights=weights)
                    loss = criterion_bce(probs, data['final_score'].reshape((batch_size, -1)).float().cuda())
                    losses+=loss
                    # Save Train loss
                    TB_looger.scalar_summary(tag=split + '_BCE-loss', value=loss, step=train_cnt)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                elif split == 'val':
                    loss = criterion_bce(probs, data['final_score'].reshape((batch_size, -1)).float().cuda())
                    losses+=loss

            # Test:
            # if split == 'test':
            pred_cls = []
            for i in range(len(pred_scores)):
                pred_cls.append(1 if pred_scores[i] > 0.5 else 0)

            f1score_class_0 = f1_score(true_scores, pred_cls, pos_label=0)
            f1score_class_1 = f1_score(true_scores, pred_cls, pos_label=1)
            f1score_global = f1_score(true_scores,pred_cls, average='macro')
            losses = losses / len(dataloaders[split])
            
            TB_looger.scalar_summary(tag=split + '_F1C1', value=f1score_class_1, step=epoch)
            log_and_print(base_logger, '\t %s Loss: %.2f F1-GLOBAL: %0.2f F1C0: %.2F F1C1: %.2f ' % (split, losses, f1score_global, f1score_class_0, f1score_class_1))

        if f1score_class_1 > rho_best:
            rho_best = f1score_class_1
            epoch_best = epoch
            name = error.split('.')[0]
            log_and_print(base_logger, '-----New best found!-----')
            if args.save:
                torch.save({'epoch': epoch,
                            'i3d': i3d.state_dict(),
                            'evaluator': evaluator.state_dict(),
                            'TSA_module': TSA_module.state_dict() if args.TSA is True else None,
                            'optimizer': optimizer.state_dict(),
                            'rho_best': rho_best},
                            f'{args.model_path}/best_{name}.pt')
            

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
    dataloaders, labels = get_dataloaders(args)

    main(dataloaders, i3d, evaluator, base_logger, TB_looger, args, labels)
