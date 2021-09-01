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

from dataset import Train_TemAug_Dataset_SHT_I3D, Test_Dataset_SHT_I3D
from model.I3D import I3D_Co
from losses import Weighted_BCE_Loss
from balanced_dataparallel import BalancedDataParallel


def eval_epoch(config, model, test_dataloader):
    model = model.eval()
    total_labels, total_scores =  [], []

    for frames, flows, ano_types, idxs, annos in tqdm(test_dataloader):
        # [42, 3, 16, 240, 320]
        # frames=frames.float().contiguous().view([-1, 3, frames.shape[-3], frames.shape[-2], frames.shape[-1]]).cuda()
        # flows=flows.float().contiguous().view([-1, 2, flows.shape[-3], flows.shape[-2], flows.shape[-1]]).cuda()
        frames = frames.float().contiguous().cuda()
        flows = flows.float().contiguous().cuda()

        with torch.no_grad():
            scores = model.module.forward_test(frames)

        for clip, score, ano_type, idx, anno in zip(frames, scores, ano_types, idxs, annos):
            score = [score.squeeze()[1].detach().cpu().float().item()] * config['segment_len']
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
    ref_dataset_nor = Train_TemAug_Dataset_SHT_I3D(config['rgb_dataset_path'], config['flow_dataset_path'], config['train_split'],
                                                config['pseudo_labels'], config['clips_num'],
                                                segment_len=config['segment_len'], type='Normal', ten_crop=config['ten_crop'], hard_label=True)

    ref_dataset_abn = Train_TemAug_Dataset_SHT_I3D(config['rgb_dataset_path'], config['flow_dataset_path'],
                                               config['train_split'],
                                               config['pseudo_labels'], config['clips_num'],
                                               segment_len=config['segment_len'], type='Abnormal',
                                               ten_crop=config['ten_crop'], hard_label=True)

    norm_dataset = Train_TemAug_Dataset_SHT_I3D(config['rgb_dataset_path'], config['flow_dataset_path'], config['train_split'],
                                                config['pseudo_labels'], config['clips_num'],
                                                segment_len=config['segment_len'], type='Normal', ten_crop=config['ten_crop'], hard_label=True)

    abnorm_dataset = Train_TemAug_Dataset_SHT_I3D(config['rgb_dataset_path'], config['flow_dataset_path'], config['train_split'],
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
    test_dataset = Test_Dataset_SHT_I3D(config['rgb_dataset_path'], config['flow_dataset_path'], config['test_split'],
                                        config['test_mask_dir'], segment_len=config['segment_len'], ten_crop=config['ten_crop'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False,
                                 num_workers=10, worker_init_fn=worker_init, drop_last=False, pin_memory=True)


    #### Model setting ####
    model = I3D_Co(config['dropout_rate'], config['expand_k'],
                        freeze_backbone=config['freeze_backbone'], freeze_blocks=config['freeze_blocks'],
                        freeze_bn=config['freeze_backbone'],freeze_bn_statics=True,
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
    criterion = Weighted_BCE_Loss(weights=config['class_reweights'],label_smoothing=config['label_smoothing'], eps=1e-8).cuda()

    # Training
    train_utils.log('Start train')
    iterator = 0
    test_epoch = 10 if config['eval_epoch'] is None else config['eval_epoch']
    AUCs,tious,best_epoch,best_tiou_epoch,best_tiou,best_AUC=[],[],0,0,0,0
    # auc = eval_epoch(config, model, test_dataloader)

    ref_iter_abn = iter(ref_dataloader_abn)
    abnorm_iter = iter(abnorm_dataloader)
    norm_iter = iter(norm_dataloader)

    for epoch in range(config['epochs']):

        if config['freeze_backbone'] and epoch == config['freeze_epochs']:
            model.module.freeze_backbone=False
            model.module.freeze_bn=False
            model.module.freeze_bn_statics=False  # 我加的
            model.module.freeze_part_model()
            model.module.freeze_batch_norm()

        Errs, Atten_Errs, Rmses = AverageMeter(), AverageMeter(), AverageMeter()
        for step, (ref_frames_nor, ref_flows_nor, ref_labels_nor) in tqdm(enumerate(ref_dataloader_nor)):
            # [B,N,(10crops),C,T,H,W]  [20,3,(10crops),3,16,224,224] ->[B*N,C,T,W,H]

            try:
                ref_frames_abn, ref_flows_abn, ref_labels_abn = next(ref_iter_abn)
            except:
                del ref_iter_abn
                ref_iter_abn = iter(ref_dataloader_abn)
                ref_frames_abn, ref_flows_abn, ref_labels_abn = next(ref_iter_abn)

            try:
                norm_frames, norm_flows, norm_labels = next(norm_iter)
            except:
                del norm_iter
                norm_iter = iter(norm_dataloader)
                norm_frames, norm_flows, norm_labels = next(norm_iter)

            try:
                abnorm_frames, abnorm_flows, abnorm_labels = next(abnorm_iter)
            except:
                del abnorm_iter
                abnorm_iter = iter(abnorm_dataloader)
                abnorm_frames, abnorm_flows, abnorm_labels = next(abnorm_iter)

            ref_frames_nor, ref_flows_nor, ref_labels_nor = ref_frames_nor.cuda().float(), ref_flows_nor.cuda().float(), ref_labels_nor.cuda().float()
            ref_frames_abn, ref_flows_abn, ref_labels_abn = ref_frames_abn.cuda().float(), ref_flows_abn.cuda().float(), ref_labels_abn.cuda().float()
            norm_frames, norm_flows = norm_frames.cuda().float(), norm_flows.cuda().float()
            abnorm_frames, abnorm_flows = abnorm_frames.cuda().float(), abnorm_flows.cuda().float()

            scores_nor, scores_p_nor, ref_attn_feat_nor, sup_attn_feat_nor = model(ref_frames_nor, ref_flows_nor, norm_frames, norm_flows, abnorm_frames, abnorm_flows)
            scores_abn, scores_p_abn, ref_attn_feat_abn, sup_attn_feat_abn = model(ref_frames_abn, ref_flows_abn, norm_frames, norm_flows, abnorm_frames, abnorm_flows, mode='Abnormal')
            scores = torch.cat([scores_nor, scores_abn], dim=0)
            scores = scores.view([-1, 2])[:, -1]
            scores_p = torch.cat([scores_p_nor, scores_p_abn], dim=0).view([-1])
            # print('scores:')
            # print(scores.shape)

            ref_labels_nor = torch.zeros_like(scores_nor).cuda().float()
            ref_labels_abn = torch.ones_like(scores_abn).cuda().float()
            labels = torch.cat([ref_labels_nor, ref_labels_abn], dim=0)
            labels = labels.view([-1, 2])[:, -1]
            # print('labels:')
            # print(labels)
            # print(labels.shape)

            ref_labels_nor_p = torch.zeros_like(scores_p_nor).cuda().float()
            ref_labels_abn_p = torch.ones_like(scores_p_abn).cuda().float()
            labels_p = torch.cat([ref_labels_nor_p, ref_labels_abn_p], dim=0).view([-1])

            err = criterion(scores, labels)
            err_p = criterion(scores_p, labels_p)

            loss = config['lambda_base'] * err + err_p

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if iterator % config['accumulate_step'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()

            rmse = cal_rmse(scores.detach().cpu().numpy(), labels.unsqueeze(-1).detach().cpu().numpy())
            Rmses.update(rmse), Errs.update(err)

            iterator += 1
        train_utils.log('[{}]: err\t{:.4f}\tatten\t{:.4f}'.format(epoch, Errs, Atten_Errs))
        Errs.reset(), Rmses.reset(), Atten_Errs.reset()

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
            model = model.train()

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
