import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import random
import time

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from apex import amp

import train_utils
from train_utils import AverageMeter
from eval_utils import eval, cal_rmse

from dataset_C3D import Train_TemAug_Dataset_SHT_C3D, Test_Dataset_SHT_C3D
from model.C3D_hard import C3D_Hard
from losses import Weighted_BCE_Loss, Hard_loss, sparsity, smooth
from balanced_dataparallel import BalancedDataParallel


def eval_epoch(config, model, test_dataloader):
    model = model.eval()
    total_labels, total_scores =  [], []

    for frames, flows, ano_types, idxs, annos in tqdm(test_dataloader):
        # [42, 3, 16, 240, 320]
        frames=frames.float().contiguous().view([-1, 1, 3, frames.shape[-3], frames.shape[-2], frames.shape[-1]]).cuda()
        flows=flows.float().contiguous().view([-1, 1, 2, flows.shape[-3], flows.shape[-2], flows.shape[-1]]).cuda()
        with torch.no_grad():
            scores = model(frames, flows, frames, flows, frames, flows, isTrain=False)
            scores = scores[1]
        for clip, score, ano_type, idx, anno in zip(frames, scores, ano_types, idxs, annos):
            score = [score.detach().cpu().float().item()] * config['segment_len']
            anno=anno.detach().numpy().astype(int)
            total_scores.extend(score)
            total_labels.extend(anno.tolist())

    return eval(total_scores,total_labels)

def train(config):
    #### set the save and log path ####
    save_path = config['save_path']
    train_utils.set_save_path(save_path)
    train_utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(config['save_path'], 'tensorboard'))
    yaml.dump(config, open(os.path.join(config['save_path'], 'classifier_config.yaml'), 'w'))

    #### make datasets ####
    def worker_init(worked_id):
        np.random.seed(worked_id)
        random.seed(worked_id)

    # train
    ref_dataset_nor = Train_TemAug_Dataset_SHT_C3D(config['rgb_dataset_path'], config['flow_dataset_path'], config['train_split'],
                                                config['pseudo_labels'], config['clips_num'],
                                                segment_len=config['segment_len'], type='Normal', ten_crop=config['ten_crop'], hard_label=True)
    ref_dataset_abn = Train_TemAug_Dataset_SHT_C3D(config['rgb_dataset_path'], config['flow_dataset_path'],
                                                  config['train_split'],
                                                  config['pseudo_labels'], config['clips_num'],
                                                  segment_len=config['segment_len'], type='Abnormal',
                                                  ten_crop=config['ten_crop'], hard_label=True)
    norm_dataset = Train_TemAug_Dataset_SHT_C3D(config['rgb_dataset_path'], config['flow_dataset_path'], config['train_split'],
                                                config['pseudo_labels'], config['clips_num'],
                                                segment_len=config['segment_len'], type='Normal', ten_crop=config['ten_crop'], hard_label=True)
    abnorm_dataset = Train_TemAug_Dataset_SHT_C3D(config['rgb_dataset_path'], config['flow_dataset_path'], config['train_split'],
                                                  config['pseudo_labels'], config['clips_num'],
                                                  segment_len=config['segment_len'], type='Abnormal', ten_crop=config['ten_crop'], hard_label=True)

    ref_dataloader_nor = DataLoader(ref_dataset_nor, batch_size=config['batch_size'], shuffle=True,
                                 num_workers=5, worker_init_fn=worker_init, drop_last=True, pin_memory=True)
    ref_dataloader_abn = DataLoader(ref_dataset_abn, batch_size=config['batch_size'], shuffle=True,
                                    num_workers=5, worker_init_fn=worker_init, drop_last=True, pin_memory=True)
    norm_dataloader = DataLoader(norm_dataset, batch_size=config['batch_size'], shuffle=True,
                                 num_workers=5, worker_init_fn=worker_init, drop_last=True, pin_memory=True)
    abnorm_dataloader = DataLoader(abnorm_dataset, batch_size=config['batch_size'], shuffle=True,
                                   num_workers=5, worker_init_fn=worker_init, drop_last=True, pin_memory=True)

    # test
    test_dataset = Test_Dataset_SHT_C3D(config['rgb_dataset_path'], config['flow_dataset_path'], config['test_split'],
                                        config['test_mask_dir'], segment_len=config['segment_len'], ten_crop=config['ten_crop'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False,
                                 num_workers=10, worker_init_fn=worker_init, drop_last=False, pin_memory=True)

    #### Model setting ####
    model = C3D_Hard(config['feature_dim'], config['dropout_rate'],
                        freeze_backbone=config['freeze_backbone'], freeze_blocks=config['freeze_blocks'],
                        pretrained_backbone=config['pretrained'], rgb_model_path=config['rgb_pretrained_model_path'],flow_model_path=config['flow_pretrained_model_path']
                        ).cuda()

    # optimizer setting
    params = list(model.parameters())
    optimizer, lr_scheduler = train_utils.make_optimizer(
        params, config['optimizer'], config['optimizer_args'])
    lr_policy = lambda epoch: (epoch + 0.5) / (config['warmup_epochs']) \
        if epoch < config['warmup_epochs'] else 1
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_policy)

    opt_level = 'O1'
    amp.init(allow_banned=True)
    amp.register_float_function(torch, 'softmax')
    amp.register_float_function(torch, 'sigmoid')
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level, keep_batchnorm_fp32=None)
    if config['ten_crop']:
        model = BalancedDataParallel(int(config['batch_size'] * 2 * config['clips_num'] * 10 / len(config['gpu']) * config['gpu0sz']), model, dim=0,
                                     device_ids=config['gpu'])
    else:
        model = BalancedDataParallel(
                                    int(config['batch_size'] * 2 * config['clips_num'] / len(config['gpu']) * config['gpu0sz']), model, dim=0,
                                    device_ids=config['gpu'])
        # model = torch.nn.parallel.DataParallel(model, dim=0, device_ids=config['gpu'])
    model = model.train()
    # criterion = Weighted_BCE_Loss(weights=config['class_reweights'],label_smoothing=config['label_smoothing'], eps=1e-8).cuda()
    criterion = Hard_loss(config)

    # Training
    train_utils.log('Start train')
    iterator = 0
    test_epoch = 10 if config['eval_epoch'] is None else config['eval_epoch']
    AUCs,tious,best_epoch,best_tiou_epoch,best_tiou,best_AUC=[],[],0,0,0,0

    ref_iter_abn = iter(ref_dataloader_abn)
    sup_iter_nor = iter(norm_dataloader)
    sup_iter_abn = iter(abnorm_dataloader)

    # auc = eval_epoch(config, model, test_dataloader)
    # print(auc)

    for epoch in range(config['epochs']):
        model = model.train()
        # if config['freeze_backbone'] and epoch == config['freeze_epochs']:
        #     model.module.freeze_backbone=False
        #     model.module.freeze_bn=False
        #     model.module.freeze_bn_statics=False  # 我加的
        #     model.module.freeze_part_model()
        #     model.module.freeze_batch_norm()

        Errs, Atten_Errs, Rmses = AverageMeter(), AverageMeter(), AverageMeter()
        for step, (ref_rgb_nor, ref_flow_nor, ref_labels_nor) in tqdm(enumerate(ref_dataloader_nor)):
            # [B,N,(10crops),C,T,H,W]  [20,3,(10crops),3,16,224,224]
            try:
                ref_rgb_abn, ref_flow_abn, ref_labels_abn = next(ref_iter_abn)
            except:
                del ref_iter_abn
                ref_iter_abn = iter(ref_dataloader_abn)
                ref_rgb_abn, ref_flow_abn, ref_labels_abn = next(ref_iter_abn)

            try:
                sup_rgb_nor, sup_flow_nor, sup_labels_nor = next(sup_iter_nor)
            except:
                del sup_iter_nor
                sup_iter_nor = iter(norm_dataloader)
                sup_rgb_nor, sup_flow_nor, sup_labels_nor = next(sup_iter_nor)

            try:
                sup_rgb_abn, sup_flow_abn, sup_labels_abn = next(sup_iter_abn)
            except:
                del sup_iter_abn
                sup_iter_abn = iter(abnorm_dataloader)
                sup_rgb_abn, sup_flow_abn, sup_labels_abn = next(sup_iter_abn)

            ref_rgb_nor, ref_flow_nor, ref_labels_nor = ref_rgb_nor.cuda().float(), ref_flow_nor.cuda().float(), ref_labels_nor.cuda().float()
            ref_rgb_abn, ref_flow_abn, ref_labels_abn = ref_rgb_abn.cuda().float(), ref_flow_abn.cuda().float(), ref_labels_abn.cuda().float()
            sup_rgb_nor, sup_flow_nor, sup_labels_nor = sup_rgb_nor.cuda().float(), sup_flow_nor.cuda().float(), sup_labels_nor.cuda().float()
            sup_rgb_abn, sup_flow_abn, sup_labels_abn = sup_rgb_abn.cuda().float(), sup_flow_abn.cuda().float(), sup_labels_abn.cuda().float()

            # for ref normal frames
            ref_p_scores_rgb, ref_p_scores_flow, ref_scores, ref_attn_feat, sup_attn_feat = \
                model(ref_rgb_nor, ref_flow_nor, sup_rgb_nor, sup_flow_nor, sup_rgb_abn, sup_flow_abn)
            ref_p_labels_nor_rgb = torch.zeros_like(ref_p_scores_rgb).cuda().float()
            ref_p_labels_nor_flow = torch.zeros_like(ref_p_scores_flow).cuda().float()
            ref_labels_nor = torch.zeros_like(ref_scores).cuda().float()
            cost1 = criterion(ref_scores, ref_labels_nor, ref_attn_feat, sup_attn_feat) + \
                    criterion(ref_p_scores_rgb, ref_p_labels_nor_rgb, ref_attn_feat, sup_attn_feat) + \
                    criterion(ref_p_scores_flow, ref_p_labels_nor_flow, ref_attn_feat, sup_attn_feat)
            with amp.scale_loss(cost1, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # if iterator % config['accumulate_step'] == 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            #     optimizer.step()
            #     optimizer.zero_grad()

            # for ref abnormal frames
            ref_p_scores_rgb, ref_p_scores_flow, ref_scores, ref_attn_feat, sup_attn_feat = \
                model(ref_rgb_abn, ref_flow_abn, sup_rgb_nor, sup_flow_nor, sup_rgb_abn, sup_flow_abn, mode='abnormal')
            ref_p_labels_abn_rgb = torch.ones_like(ref_p_scores_rgb).cuda().float()
            ref_p_labels_abn_flow = torch.ones_like(ref_p_scores_flow).cuda().float()
            ref_labels_abn = torch.ones_like(ref_scores).cuda().float()

            loss_sparse = sparsity(ref_p_scores_rgb, config['batch_size'], 8e-3)  # 对T个伪异常帧进行稀疏化，因为这其中含有大量的正常帧
            loss_smooth = smooth(ref_p_scores_rgb, 8e-4)
            cost2 = criterion(ref_scores, ref_labels_abn, ref_attn_feat, sup_attn_feat) + \
                    criterion(ref_p_scores_rgb, ref_p_labels_abn_rgb, ref_attn_feat, sup_attn_feat) + \
                    criterion(ref_p_scores_flow, ref_p_labels_abn_flow, ref_attn_feat, sup_attn_feat)
                    # loss_smooth + loss_sparse + \

            with amp.scale_loss(cost2, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            # if iterator % config['accumulate_step'] == 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            #     optimizer.step()
            #     optimizer.zero_grad()

            Errs.update(cost1+cost2)

            iterator += 1
        train_utils.log('[{}]: err\t{:.4f}\tatten\t{:.4f}'.format(epoch, Errs, Atten_Errs))
        Errs.reset(), Atten_Errs.reset()

        lr_scheduler.step()
        train_utils.log("epoch {}, lr {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_utils.log('----------------------------------------')

        if epoch % test_epoch == 0 and epoch > 20:

            auc = eval_epoch(config, model, test_dataloader)
            AUCs.append(auc)
            if len(AUCs) >= 5:
                mean_auc = sum(AUCs[-5:]) / 5.
                if mean_auc > best_AUC:
                    best_epoch,best_AUC =epoch,mean_auc
                train_utils.log('best_AUC {} at epoch {}, now {}'.format(best_AUC, best_epoch, mean_auc))

            train_utils.log('===================')
            if auc > 0.8:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(save_path, 'models/model-epoch-{}-AUC-{}.pth'.format(epoch, auc)))

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
