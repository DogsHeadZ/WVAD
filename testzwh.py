# import time
import torch
# import numpy as np
# import random
# from model.modules import Relation
# ralation = Relation(feat_dim=2048)
# input1 = torch.rand([12,2048])
# input2 = torch.rand([10,2048])
#
# out = ralation.attention_module_multi_head(input1, input2)
# print(out.shape)
#
#
# from model.modules import VideoRelation
# vralation = VideoRelation(feat_dim=2048)
# input1 = torch.rand([3,12,2048])
# input2 = torch.rand([3,10,2048])
#
# out = vralation.attention_module_multi_head(input1, input2)
# print(out.shape)
a = torch.randint(1,10,(2,2,3,2)).float()
print(torch.max(a))

# a = torch.randint(1,10,(2,2,3,2)).float()
# b = torch.randint(1,10,(2,2,3,2)).float()
# print(a)
# print(b)
# result = torch.matmul(a, b.permute(0, 1,3, 2))
# print(result)

# begin = time.time()
# time.sleep(1)
# end  = time.time()
# print(end-begin)
#
# import h5py
#
# begin = time.time()
#
# rgb_h5 = h5py.File('../AllDatasets/SHT_Frames.h5', 'r')
#
#
# a = rgb_h5['01_0014' + '-{0:06d}'.format(0)][:]
# end  = time.time()
# print(end-begin)

#
# abn_score = torch.Tensor([0,0.1,0.9,1,0.3,1])
# abn_rgb_feat = torch.Tensor([[0,1,2,3],
#                              [1,2,3,4],
#                              [2,3,4,5],
#                              [3,4,5,6],
#                              [4,5,6,7],
#                              [5,6,7,8]])
#
# topK = 2
# threshold = 0.4
# abn_score_nor = abn_score[torch.where(abn_score<threshold)]
# abn_rgb_feat_nor = abn_rgb_feat[torch.where(abn_score<threshold)]
# if len(abn_score_nor) > topK:
#     abn_score_nor, abn_k_idx = torch.topk(abn_score_nor, topK, dim=0, largest=False)
#     abn_rgb_feat_nor = abn_rgb_feat_nor[abn_k_idx]
#
# # print(abn_score_nor)
# # print(abn_rgb_feat_nor)
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True


# 设置随机数种子
# setup_seed(20)
#
# a = torch.randint(1,10,(2,3)).float()
# print(a)
# b = torch.randint(1,10,(2,3,5)).float()
# print(b)
#
# abn_k_nor_idx = torch.topk(a , 2, dim=1, largest=False)[1]
# print(abn_k_nor_idx)
# print(abn_k_nor_idx.shape)
#
# abn_k_nor_idx = abn_k_nor_idx.unsqueeze(2).expand([-1, -1, b.shape[2]])
# print(abn_k_nor_idx)
#
# c = torch.gather(b, 1, abn_k_nor_idx)
# print(c)

# fusion_rgb_feat = torch.cat([a, b], dim=2)
# print(fusion_rgb_feat.shape)
#
# inter_sim = torch.matmul(a,b.permute(0, 2, 1))
# print(inter_sim)
# inter_sim_mean = torch.mean(inter_sim, dim=2)
#
# print(inter_sim_mean)
# print(inter_sim_mean.shape)

# ref_flow_feat = torch.randint(1,10,(2,3,10)).float()
# print(ref_flow_feat)
# norm = torch.norm(ref_flow_feat, p=2, dim=2).unsqueeze(2)
# print(norm)
# xx = ref_flow_feat/norm
# print(xx)
# print(xx.shape)
# print('module.Regressor.'.__len__())

# labels = np.array([1., 1., 1., 0., 0.], dtype=np.float32)
# scores = np.array([0.1, 0.15, 0.2, 0.05, 0.002], dtype=np.float32)
# labels=labels.astype(bool)
# print(labels)
# print(scores[labels])
# neg_labels=(1-labels).astype(bool)
# positive=np.mean(scores[labels])
# negative=np.mean(scores[neg_labels])
# print(positive-negative)
#
# labels = []
# label = np.array([1., 0.], dtype=np.float32)
# labels.append(label)
#
# labels.append(label)
# labels.append(label)
# print(np.array(labels))
#
# features = np.load('data/rtfm/list/SH_Test_ten_crop_i3d/01_0015_i3d.npy')
# print(features.shape)
# labels  = np.load('../AllDatasets/ShanghaiTech/test_frame_mask/01_0015.npy')
# print(labels.shape)

# a = torch.randint(1,10,(2,10)).float()
# feat = torch.randint(1,10,(2,10,5)).float()
# print(a)
# b = torch.abs_(a-5)
# print(b)
# topK = 6
# k_idx = torch.topk(b, topK, dim=1)[1]
# ori = torch.gather(a, 1, k_idx)
# print(k_idx)
# print('features:')
# print(feat)
# print('scores:')
# print(ori)
# print('hard features:')
# hard_feat =  torch.gather(feat, 1, k_idx.unsqueeze(2).expand([-1, -1, 5]))
# print(hard_feat)
#
# topK=2
# hard_topK_idx_abn = torch.topk(ori, topK, dim=1)[1]
# hard_topK_feat_abn = torch.gather(hard_feat, 1, hard_topK_idx_abn.unsqueeze(2).expand([-1, -1, 5]))
# print(hard_topK_feat_abn)

# print(torch.gather(hard_feat, 1, normal_idx.unsqueeze(2).expand([-1, -1, 5])))
# k_idx = k_idx.unsqueeze(2).expand([-1, -1, 5])
# c = torch.gather(feat, 1, k_idx)
# print(feat)
# print(c)


