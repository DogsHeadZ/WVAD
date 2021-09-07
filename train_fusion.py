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
from train_utils import AverageMeter
from eval_utils import eval, cal_rmse

from dataset_feat import Dataset_f
# from dataset import Train_TemAug_Dataset_SHT_I3D, Test_Dataset_SHT_I3D
from model.rtfm_model import CoModel
from losses import Weighted_BCE_Loss
from balanced_dataparallel import BalancedDataParallel


def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
    fo.write(str(test_info["test_AUC"][-1]))
    fo.close()


def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()

    def forward(self, ref_scores, ref_labels):

        labels = ref_labels.squeeze()
        scores = ref_scores.squeeze()
        loss_cls = self.criterion(scores, labels)  # BCE loss in the score space
        loss_total = loss_cls

        return loss_total


def eval_epoch(config, model, test_dataloader):


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
            # print(labels[0].shape)
            preds = torch.cat((preds, logits))
            gt_labels = torch.cat((gt_labels, labels[0]))

        gt = np.load(config['gt'])

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
    def worker_init(worked_id):
        np.random.seed(worked_id)
        random.seed(worked_id)

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

    model = CoModel(config['feature_size'], config['batch_size']).cuda()


    # optimizer setting
    params = list(model.parameters())
    optimizer, lr_scheduler = train_utils.make_optimizer(
        params, config['optimizer'], config['optimizer_args'])


    # opt_level = 'O1'
    # amp.init(allow_banned=True)
    # amp.register_float_function(torch, 'softmax')
    # amp.register_float_function(torch, 'sigmoid')
    # model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level, keep_batchnorm_fp32=None)
    # if config['ten_crop']:
    #     model = BalancedDataParallel(int(config['batch_size'] * 2 * config['clips_num'] * 10 / len(config['gpu']) * config['gpu0sz']), model, dim=0,
    #                                  device_ids=config['gpu'])
    # else:
    #     model = BalancedDataParallel(
    #                                 int(config['batch_size'] * 2 * config['clips_num'] / len(config['gpu']) * config['gpu0sz']), model, dim=0,
    #                                 device_ids=config['gpu'])
    #     # model = torch.nn.parallel.DataParallel(model, dim=0, device_ids=config['gpu'])

    model = model.train()
    # Training
    train_utils.log('Start train')
    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    test_epoch = 5 if config['eval_epoch'] is None else config['eval_epoch']
    auc = eval_epoch(config, model, test_dataloader)

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
            ref_p_scores, ref_scores = model(ref_rgb_nor, ref_flow_nor, norm_frames, norm_flows, abn_frames, abn_flows)  # b*32  x 2048
            loss_criterion = RTFM_loss(0.0001, 100)  # 这里就只用到了topk个帧进行分类损失
            ref_p_labels_nor = torch.zeros_like(ref_p_scores).cuda().float()
            ref_labels_nor = torch.zeros_like(ref_scores).cuda().float()
            cost = loss_criterion(ref_scores, ref_labels_nor) + loss_criterion(ref_p_scores, ref_p_labels_nor)
            # cost = loss_criterion(ref_p_scores, ref_p_labels_nor)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            ref_p_scores, ref_scores = model(ref_rgb_abn, ref_flow_abn, norm_frames, norm_flows, abn_frames, abn_flows,mode='abnormal')  # b*32  x 2048
            loss_criterion = RTFM_loss(0.0001, 100)  # 这里就只用到了topk个帧进行分类损失
            loss_sparse = sparsity(ref_p_scores, config['batch_size'], 8e-3)  # 对T个伪异常帧进行稀疏化，因为这其中含有大量的正常帧
            loss_smooth = smooth(ref_p_scores, 8e-4)

            ref_p_labels_abn = torch.ones_like(ref_p_scores).cuda().float()

            ref_labels_abn = torch.ones_like(ref_scores).cuda().float()
            cost = loss_criterion(ref_scores, ref_labels_abn) + loss_criterion(ref_p_scores, ref_p_labels_abn) + loss_smooth + loss_sparse
            # cost = loss_criterion(ref_p_scores, ref_p_labels_abn) + loss_smooth + loss_sparse

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        if step % test_epoch == 0 and step > 200:

            auc = eval_epoch(config, model, test_dataloader)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)
            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                # torch.save(model.state_dict(), './ckpt/' + args.model_name + '{}-i3d.pkl'.format(step))
                torch.save(model.state_dict(), os.path.join(save_path, 'models/model-step-{}.pth'.format(step)))
                save_best_record(test_info, os.path.join(save_path, '{}-step-AUC.txt'.format(step)))

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
