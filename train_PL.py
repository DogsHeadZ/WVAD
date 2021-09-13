import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import random
import time
from sklearn.metrics import auc, roc_curve, precision_recall_curve

import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from apex import amp

import train_utils

from dataset_feat import Dataset_f
# from dataset import Train_TemAug_Dataset_SHT_I3D, Test_Dataset_SHT_I3D
from model.model_hard import HardModel
from losses import RTFM_loss, sparsity, smooth
from balanced_dataparallel import BalancedDataParallel


def eval_epoch(model, test_dataloader):

    with torch.no_grad():
        model.eval()
        preds = torch.zeros(0)
        gt_labels = torch.zeros(0)
        for i, (frames, flows, labels) in enumerate(test_dataloader):
            frames, flows, labels = frames.cuda(), flows.cuda(), labels.cuda()
            frames = frames.permute(0,2,1,3) # [bs, 10, T, F]

            ref_scores = model(frames, frames, frames,frames,frames,frames)  # 他这个注意力机制一次要输入一整个视频，而不能一帧一帧进行检测
            logits = ref_scores[0]
            # print(logits.shape)
            # print(labels.shape)

            preds = torch.cat((preds, logits))
            gt_labels = torch.cat((gt_labels, labels[0]))

        # gt = np.load(config['gt'])

        preds = list(preds.cpu().detach().numpy())
        preds = np.repeat(np.array(preds), 16)

        gt_labels = gt_labels.cpu().detach().numpy()

        fpr, tpr, threshold = roc_curve(gt_labels, preds)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        return rec_auc

def train(config):
    #### set the save and log path ####
    save_path = config['save_path']
    train_utils.set_save_path(save_path)
    train_utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(config['save_path'], 'tensorboard'))
    yaml.dump(config, open(os.path.join(config['save_path'], 'classifier_config.yaml'), 'w'))

    #### make datasets ####
    # train
    ref_dataloader_nor = DataLoader(Dataset_f(config['rgb_list'], config['rgb_file_path'], config['flow_file_path'], config['train_split'],
                                              config['test_split'], config['test_mask_dir'], test_mode=False, is_normal=True),
                                    batch_size=config['batch_size'], shuffle=True,
                                    num_workers=0, pin_memory=False, drop_last=True)
    ref_dataloader_abn = DataLoader(Dataset_f(config['rgb_list'], config['rgb_file_path'], config['flow_file_path'], config['train_split'],
                                              config['test_split'], config['test_mask_dir'], test_mode=False, is_normal=False),
                                    batch_size=config['batch_size'], shuffle=True,
                                    num_workers=0, pin_memory=False, drop_last=True)

    norm_dataloader = DataLoader(Dataset_f(config['rgb_list'], config['rgb_file_path'], config['flow_file_path'], config['train_split'],
                                           config['test_split'], config['test_mask_dir'], test_mode=False, is_normal=True),
                                 batch_size=config['batch_size'], shuffle=True,
                                 num_workers=0, pin_memory=False, drop_last=True)
    abnorm_dataloader = DataLoader(Dataset_f(config['rgb_list'], config['rgb_file_path'], config['flow_file_path'], config['train_split'],
                                             config['test_split'], config['test_mask_dir'], test_mode=False, is_normal=False),
                                   batch_size=config['batch_size'], shuffle=True,
                                   num_workers=0, pin_memory=False, drop_last=True)

    test_dataloader = DataLoader(Dataset_f(config['test_rgb_list'], config['rgb_file_path'], config['flow_file_path'], config['train_split'],
                                           config['test_split'], config['test_mask_dir'], test_mode=True),
                             batch_size=1, shuffle=False,  ####
                             num_workers=0, pin_memory=False)

    model = HardModel(config['feature_size'], train_PL=True).cuda()


    # optimizer setting
    params = list(model.parameters())
    optimizer, lr_scheduler = train_utils.make_optimizer(
        params, config['optimizer'], config['optimizer_args'])


    # opt_level = 'O1'
    # amp.init(allow_banned=True)
    # amp.register_float_function(torch, 'softmax')
    # amp.register_float_function(torch, 'sigmoid')
    # model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level, keep_batchnorm_fp32=None)

    model = torch.nn.parallel.DataParallel(model, dim=0, device_ids=config['gpu'])

    model = model.train()
    # Training
    train_utils.log('Start train')
    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    test_epoch = 5 if config['eval_epoch'] is None else config['eval_epoch']
    auc = eval_epoch(model, test_dataloader)
    loss_f = torch.nn.BCELoss()

    for step in tqdm(
            range(1, config['max_epoch'] + 1),
            total=config['max_epoch'],
            dynamic_ncols=True
    ):

        if (step - 1) % len(ref_dataloader_nor) == 0:
            ref_iter_nor = iter(ref_dataloader_nor)
        if (step - 1) % len(ref_dataloader_abn) == 0:
            ref_iter_abn = iter(ref_dataloader_abn)
        if (step - 1) % len(norm_dataloader) == 0:
            nor_iter = iter(norm_dataloader)
        if (step - 1) % len(abnorm_dataloader) == 0:
            abn_iter = iter(abnorm_dataloader)

        ref_rgb_nor, ref_flow_nor, ref_labels_nor = next(ref_iter_nor)
        ref_rgb_nor, ref_flow_nor, ref_labels_nor = ref_rgb_nor.cuda().float(), ref_flow_nor.cuda().float(), ref_labels_nor.cuda().float()

        ref_rgb_abn, ref_flow_abn, ref_labels_abn = next(ref_iter_abn)
        ref_rgb_abn, ref_flow_abn, ref_labels_abn = ref_rgb_abn.cuda().float(), ref_flow_abn.cuda().float(), ref_labels_abn.cuda().float()

        norm_frames, norm_flows, norm_labels = next(nor_iter)
        norm_frames, norm_flows, norm_labels = norm_frames.cuda().float(), norm_flows.cuda().float(), norm_labels.cuda().float()

        abn_frames, abn_flows, abn_labels = next(abn_iter)
        abn_frames, abn_flows, abn_labels = abn_frames.cuda().float(), abn_flows.cuda().float(), abn_labels.cuda().float()

        with torch.set_grad_enabled(True):
            model.train()
            ref_p_scores_rgb, ref_p_scores_flow = model(ref_rgb_nor, ref_flow_nor, norm_frames, norm_flows, abn_frames, abn_flows)  # b*32  x 2048
            loss_criterion = RTFM_loss(0.0001, 100)  # 这里就只用到了topk个帧进行分类损失

            ref_p_labels_nor_rgb = torch.zeros_like(ref_p_scores_rgb).cuda().float()
            ref_p_labels_nor_flow = torch.zeros_like(ref_p_scores_flow).cuda().float()

            cost1 = loss_f(ref_p_scores_rgb, ref_p_labels_nor_rgb) + loss_f(ref_p_scores_flow, ref_p_labels_nor_flow)

            ref_p_scores_rgb, ref_p_scores_flow = model(ref_rgb_abn, ref_flow_abn, norm_frames, norm_flows, abn_frames, abn_flows,mode='abnormal')  # b*32  x 2048
            loss_criterion = RTFM_loss(0.0001, 100)  # 这里就只用到了topk个帧进行分类损失
            loss_sparse = sparsity(ref_p_scores_rgb, config['batch_size'], 8e-3)  # 对T个伪异常帧进行稀疏化，因为这其中含有大量的正常帧
            loss_smooth = smooth(ref_p_scores_rgb, 8e-4)
            ref_p_labels_abn_rgb = torch.ones_like(ref_p_scores_rgb).cuda().float()
            ref_p_labels_abn_flow = torch.ones_like(ref_p_scores_flow).cuda().float()
            cost2 = loss_f(ref_p_scores_rgb, ref_p_labels_abn_rgb) + \
                    loss_f(ref_p_scores_flow, ref_p_labels_abn_flow) + \
                    loss_smooth + loss_sparse

            cost = cost1 + cost2
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # with amp.scale_loss(cost, optimizer) as cost:
            #     optimizer.zero_grad()
            #     cost.backward()
            #     optimizer.step()

        if step % test_epoch == 0 and step > 50:

            auc = eval_epoch(model, test_dataloader)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)
            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                # torch.save(model.state_dict(), './ckpt/' + args.model_name + '{}-i3d.pkl'.format(step))
                torch.save(model.state_dict(), os.path.join(save_path, 'models/model-step-{}.pth'.format(step)))
                train_utils.save_best_record(test_info, os.path.join(save_path, '{}-step-AUC.txt'.format(step)))

    # torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')
    torch.save(model.state_dict(), os.path.join(save_path, 'models/model-final.pth'))

    train_utils.log('Training is finished')
    train_utils.log('max_frame_AUC: {}'.format(best_AUC))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    # for flownet2, no need to modify
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    if args.tag is not None:
        config['save_path'] += ('_' + args.tag)

    train_utils.set_gpu(args.gpu)
    config['gpu'] =[i for i in range(len(args.gpu.split(',')))]

    train(config)
