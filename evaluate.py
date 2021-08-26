import os
import yaml
import argparse
import random
import numpy as np
from tqdm import tqdm
import cv2

import torch
from torch.utils.data import DataLoader
from apex import amp

import train_utils
from eval_utils import eval
from model.I3D_STD import I3D_SGA_STD
from dataset_i3d import Test_Dataset_C3D,Test_Dataset_I3D,Test_Dataset_SHT_C3D,Test_Dataset_SHT_I3D

def load_model(model,state_dict):
    new_dict={}
    for key,value in state_dict.items():
        new_dict[key[7:]]=value
    model.load_state_dict(new_dict)


def eval_UCF(config,model,test_dataloader):
    total_labels, total_scores= [], []
    data_iter=test_dataloader.__iter__()
    next_batch=data_iter.__next__()
    next_batch[0]=next_batch[0].cuda(non_blocking=True)

    for frames,ano_types, keys, idxs,annos in tqdm(test_dataloader):
        frames=frames.float().contiguous().view([-1, 3, frames.shape[-3], frames.shape[-2], frames.shape[-1]]).cuda()

        with torch.no_grad():
            scores, feat_maps = model(frames)[:2]

        if config['ten_crop']:
            scores = scores.view([-1, 10, 2]).mean(dim=-2)
        for i, (clip, score, ano_type, key, idx, anno, feat_map) in enumerate(
                zip(frames, scores, ano_types, keys, idxs, annos,
                    feat_maps)):
            anno = anno.numpy().astype(int)
            if np.isnan(anno.any()):
                raise ValueError('NaN in anno')
            score = score.float().squeeze()[1].detach().cpu().item()
            if np.isnan(score):
                raise ValueError('NaN in score')
            anno = anno.astype(int)
            score = [score] * config['segment_len']
            total_scores.extend(score)
            total_labels.extend(anno.tolist())

    return eval(total_scores, total_labels)


def eval_SHT(config, model,test_dataloader):
    total_labels, total_scores = [], []
    for frames,ano_type,_,annos in tqdm(test_dataloader):
        frames=frames.float().contiguous().view([-1, 3, frames.shape[-3], frames.shape[-2], frames.shape[-1]]).cuda()
        with torch.no_grad():
            scores, feat_maps = model(frames)[:2]
        if config['ten_crop']:
            scores = scores.view([-1, 10, 2]).mean(dim=-2)

        for clip, score, anno in zip(frames, scores, annos):
            score = [score.squeeze()[1].detach().cpu().item()] * config['segment_len']
            total_scores.extend(score)
            total_labels.extend(anno.tolist())

    return eval(total_scores, total_labels)


def test(config):
    def worker_init(worked_id):
        np.random.seed(worked_id)
        random.seed(worked_id)

    test_dataset = Test_Dataset_SHT_I3D(config['dataset_path'], config['test_split'],
                                        config['test_mask_dir'], segment_len=config['segment_len'], ten_crop=config['ten_crop'])
    test_dataloader = DataLoader(test_dataset, batch_size=42, shuffle=False, num_workers=5,
                              worker_init_fn=worker_init, drop_last=False)

    #### Model setting ####
    model = I3D_SGA_STD(config['dropout_rate'], config['expand_k'],
                        freeze_backbone=False, freeze_blocks=None).cuda().eval()
    opt_level = 'O1'
    amp.init(allow_banned=True)
    optimizer, lr_scheduler = train_utils.make_optimizer(
        model.parameters(), config['optimizer'], config['optimizer_args'])
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level, keep_batchnorm_fp32=None)

    load_model(model, torch.load(config['stored_path'])['model'])

    if config['dataset_name']=='UCF':
        eval_UCF(config,model,test_dataloader)
    else:
        eval_SHT(config, model, test_dataloader)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')

    # for flownet2, no need to modify
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    if args.tag is not None:
        config['save_path'] += ('_' + args.tag)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu
    else:
        torch.cuda.set_device(int(args.gpu))
    train_utils.set_gpu(args.gpu)

    test(config)
