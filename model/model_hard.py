import torch
import torch.nn as nn
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from model.modules_hard import Hard_sim_sample_generator, Hard_score_sample_generator, VideoRelation, Aggregate

# 这个文件是给RTFM的

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Classifier(nn.Module):
    def __init__(self,feature_dim,dropout_rate=0.7):
        super(Classifier, self).__init__()
        self.classifier=nn.Sequential(nn.Linear(feature_dim,512), nn.ReLU(), nn.Dropout(dropout_rate),
                                     nn.Linear(512,128), nn.ReLU(), nn.Dropout(dropout_rate),
                                     nn.Linear(128,1), nn.Sigmoid())

    def forward(self, x):
        # （bs * ncrops, T, F)
        scores = self.classifier(x)
        return scores


class HardModel(nn.Module):
    def __init__(self, feature_dim, train_PL=False):
        super(HardModel, self).__init__()
        self.train_PL = train_PL
        self.Aggregate = Aggregate(feat_dim=feature_dim)
        self.drop_out = nn.Dropout(0.7)

        self.p_classifier_rgb = Classifier(feature_dim=feature_dim)
        self.p_classifier_flow = Classifier(feature_dim=feature_dim)
        self.f_classifier = Classifier(feature_dim=feature_dim)

        self.hard_sim_sampler = Hard_sim_sample_generator()
        self.hard_score_sampler = Hard_score_sample_generator()
        self.relation = VideoRelation(feat_dim=feature_dim)
        self.relation_two = VideoRelation(feat_dim=feature_dim)


        self.apply(weight_init)

    def forward(self, ref_rgb, ref_flow, normal_rgb, normal_flow, abnormal_rgb, abnormal_flow, mode='normal'):

        # ref_rgb: [bs, ncrops, T, F]
        inputs_rgb = torch.cat((ref_rgb, normal_rgb, abnormal_rgb), 0)
        inputs_flow = torch.cat((ref_flow, normal_flow, abnormal_flow), 0)
        bs, ncrops, T, F = inputs_rgb.size()
        inputs_rgb = inputs_rgb.view(-1, T, F)
        inputs_flow = inputs_flow.view(-1, T, F)
        with torch.no_grad():
            p_scores_rgb = self.p_classifier_rgb(inputs_rgb).squeeze(2)  # (bs*ncrops, T)
            p_scores_flow = self.p_classifier_flow(inputs_flow).squeeze(2)  # (bs*ncrops, T)

        features = inputs_rgb
        # features = self.Aggregate(features)    #RTFM的attention放在这里精度有明显提升（能达到96.6，2个百分点），但放到后面会起很大的负作用
        # features = self.drop_out(features)  # (bs*ncrops, T, F)

        bs = bs // 3

        ref_p_scores_rgb, abn_scores_rgb = p_scores_rgb[:bs*ncrops], p_scores_rgb[2*bs*ncrops:3*bs*ncrops]
        ref_p_scores_flow, abn_scores_flow = p_scores_flow[:bs*ncrops], p_scores_flow[2*bs*ncrops:3*bs*ncrops]

        ref_rgb, ref_flow, normal_rgb, normal_flow, abnormal_rgb, abnormal_flow = \
            features[:bs*ncrops], ref_flow.view(bs*ncrops, T, -1), features[bs*ncrops:2*bs*ncrops], \
                normal_flow.view(bs*ncrops, T, -1), features[2*bs*ncrops:3*bs*ncrops], abnormal_flow.view(bs*ncrops, T, -1)

        if self.train_PL:
            ref_p_scores_rgb = ref_p_scores_rgb.view(bs, ncrops, -1).mean(1)
            ref_p_scores_flow = ref_p_scores_flow.view(bs, ncrops, -1).mean(1)
            if bs == 1:
                return ref_p_scores_rgb

            return ref_p_scores_rgb, ref_p_scores_flow

        if bs == 1:  # this is for inference
            ref_aggr = self.relation(ref_rgb, ref_rgb, index=0)  # 测试时直接与自身进行融合
            ref_aggr = self.relation_two(ref_aggr, ref_aggr, index=0)
            ref_scores = self.f_classifier(ref_aggr)
            ref_scores = ref_scores.view(bs, ncrops, -1).mean(1) # 对这个帧的10个crops取平均得到这帧的分数[bs, T]
            return ref_scores

        supn_hard_feat, supn_conf_feat = self.hard_sim_sampler(normal_rgb)
        supa_hard_nor_feat, supa_hard_abn_feat, supa_conf_nor_feat, supa_conf_abn_feat \
            = self.hard_score_sampler(abnormal_rgb, abn_scores_rgb, abn_scores_flow)

        # 第一步融合
        supn_aggr_feat = self.relation(supn_conf_feat, supn_hard_feat, index=0)  #融合正常视频困难样本的特征，使得conf样本更具有鲁棒性
        supa_aggr_nor_feat = self.relation(supa_conf_nor_feat, supa_hard_nor_feat, index=0)  # 异常视频normal clips融合
        supa_aggr_abn_feat = self.relation(supa_conf_abn_feat, supa_hard_abn_feat, index=0)  # 异常视频abnormal clips融合

        if mode == 'normal':
            ref_hard_feat, ref_conf_feat = self.hard_sim_sampler(ref_rgb)
            ref_aggr_feat = self.relation(ref_conf_feat, ref_hard_feat, index=0)
            # 第二步融合
            # 融合不同视频正常clips
            ref_aggr2_feat = self.relation_two(ref_aggr_feat, torch.cat([supn_aggr_feat, supa_aggr_nor_feat], dim=1), index=0)
            # ref_aggr2_feat = torch.cat([ref_aggr_feat, torch.cat([supn_aggr_feat, supa_aggr_nor_feat], dim=1)], dim=1)

            # 对于异常clips第二步融合就自己与自己融，与正常clips的特征构建Rank loss
            supa_aggr2_abn_feat = self.relation_two(supa_aggr_abn_feat, supa_aggr_abn_feat, index=0)

            ref_scores = self.f_classifier(ref_aggr2_feat)
            ref_p_scores_rgb = ref_p_scores_rgb.view(bs, ncrops, -1).mean(1)
            ref_p_scores_flow = ref_p_scores_flow.view(bs, ncrops, -1).mean(1)

            ref_scores = ref_scores.view(bs, ncrops, -1).mean(1)
            return ref_p_scores_rgb, ref_p_scores_flow, ref_scores, ref_aggr2_feat, supa_aggr2_abn_feat

        else:
            ref_hard_nor_feat, ref_hard_abn_feat, ref_conf_nor_feat, ref_conf_abn_feat \
                = self.hard_score_sampler(ref_rgb, ref_p_scores_rgb, ref_p_scores_flow)
            ref_aggr_nor_feat = self.relation(ref_conf_nor_feat, ref_hard_nor_feat, index=0)
            ref_aggr_abn_feat = self.relation(ref_conf_abn_feat, ref_hard_abn_feat, index=0)

            #第二步融合
            ref_aggr2_abn_feat = self.relation_two(ref_aggr_abn_feat, supa_aggr_abn_feat, index=0)
            # ref_aggr2_abn_feat = torch.cat([ref_aggr_abn_feat, supa_aggr_abn_feat], dim=1)

            ref_aggr2_nor_feat = self.relation_two(ref_aggr_nor_feat, torch.cat([supn_aggr_feat, supa_aggr_nor_feat], dim=1), index=0)

            ref_scores = self.f_classifier(ref_aggr2_abn_feat)
            ref_p_scores_rgb = ref_p_scores_rgb.view(bs, ncrops, -1).mean(1)
            ref_p_scores_flow = ref_p_scores_flow.view(bs, ncrops, -1).mean(1)

            ref_scores = ref_scores.view(bs, ncrops, -1).mean(1)
            return ref_p_scores_rgb, ref_p_scores_flow, ref_scores, ref_aggr2_nor_feat, ref_aggr2_abn_feat
