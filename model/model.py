import torch
import torch.nn as nn
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from model.modules import Fusion, MotionMemory, SA


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


# https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.4.1_to_1.1.0/lib/non_local_gaussian.py
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)



class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1,dilation=1, padding=1),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1,
                      stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(512, sub_sample=False, bn_layer=True)


    def forward(self, x):
            # x: (B, T, F)
            out = x.permute(0, 2, 1)    # 做了个转置是为了在时间维度上应用attention
            residual = out

            out1 = self.conv_1(out)
            out2 = self.conv_2(out)

            out3 = self.conv_3(out)
            out_d = torch.cat((out1, out2, out3), dim = 1)
            out = self.conv_4(out)
            out = self.non_local(out)
            out = torch.cat((out_d, out), dim=1)
            out = self.conv_5(out)   # fuse all the features together
            out = out + residual
            out = out.permute(0, 2, 1)
            # out: (B, T, F)

            return out


class FA(nn.Module):
    def __init__(self, len_feature):
        super(FA, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature

        self.non_local = NONLocalBlock1D(len_feature, sub_sample=False, bn_layer=True)


    def forward(self, x):
            # x: (B, T, F)
            out = x.permute(0, 2, 1)    # 做了个转置是为了在时间维度上应用attention
            out = self.non_local(out)
            out = out.permute(0, 2, 1)
            # out: (B, T, F)

            return out


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
    def __init__(self, feature_dim):
        super(HardModel, self).__init__()

        self.thres = 0.2

        # self.Aggregate = Aggregate(len_feature=feature_dim)
        self.Aggregate = SA(feat_dim=feature_dim)

        self.drop_out = nn.Dropout(0.7)
        self.p_classifier_rgb = Classifier(feature_dim=feature_dim)
        self.p_classifier_flow = Classifier(feature_dim=feature_dim)

        self.f_classifier = Classifier(feature_dim=feature_dim)
        self.fusion = Fusion()

        self.apply(weight_init)


    def forward(self, ref_rgb, ref_flow, normal_rgb, normal_flow, abnormal_rgb, abnormal_flow, mode='normal', tencrop=True):

        inputs_rgb = torch.cat((ref_rgb, normal_rgb, abnormal_rgb), 0)
        inputs_flow = torch.cat((ref_flow, normal_flow, abnormal_flow), 0)
        bs, ncrops, T, F = inputs_rgb.size()
        inputs_rgb = inputs_rgb.view(-1, T, F)
        inputs_flow = inputs_flow.view(-1, T, F)

        p_scores_rgb = self.p_classifier_rgb(inputs_rgb).squeeze(2)  # (bs*ncrops, T)
        p_scores_flow = self.p_classifier_flow(inputs_flow).squeeze(2)  # (bs*ncrops, T)

        features = inputs_rgb
        # features = self.Aggregate(features)    #attention放在这里精度有明显提升（能达到96.6，2个百分点），但放到后面会起很大的负作用
        # features = self.drop_out(features)  # (bs*ncrops, T, F)

        bs = bs // 3
        if tencrop:
            ref_p_scores_rgb, abn_scores_rgb = p_scores_rgb[:bs*ncrops], p_scores_rgb[2*bs*ncrops:3*bs*ncrops]
            ref_p_scores_flow, abn_scores_flow = p_scores_flow[:bs*ncrops], p_scores_flow[2*bs*ncrops:3*bs*ncrops]

            ref_rgb, ref_flow, normal_rgb, normal_flow, abnormal_rgb, abnormal_flow = \
                features[:bs*ncrops], ref_flow.view(bs*ncrops, T, -1), features[bs*ncrops:2*bs*ncrops], \
                    normal_flow.view(bs*ncrops, T, -1), features[2*bs*ncrops:3*bs*ncrops], abnormal_flow.view(bs*ncrops, T, -1)

        if bs == 1:  # this is for inference
            # ref_rgb = self.Aggregate(ref_rgb,ref_rgb)
            # ref_rgb = self.drop_out(ref_rgb)
            ref_scores = self.f_classifier(ref_rgb)
            ref_scores = ref_scores.view(bs, ncrops, -1).mean(1) # 对这个帧的10个crops取平均得到这帧的分数[bs, T]
            return ref_scores

        if mode == 'normal':
            # 当ref为normal时
            hard_topK = 10  #挑选出abnormal video中的x个困难样本，即异常分数值接近于0.5
            hard_topK_idx = torch.topk(torch.abs_(abn_scores_rgb-0.5), hard_topK, dim=1, largest=False)[1]
            hard_topK_flow_scores = torch.gather(abn_scores_flow, 1, hard_topK_idx)
            hard_topK_feat = torch.gather(abnormal_rgb, 1, hard_topK_idx.unsqueeze(2).expand([-1, -1, F]))
            hard_topK_flow_feat = torch.gather(abnormal_flow, 1, hard_topK_idx.unsqueeze(2).expand([-1, -1, F]))

            hard_topK_abn = 1   # 从x个困难样本中挑选出x个flow score为异常的样本
            hard_topK_idx_abn = torch.topk(hard_topK_flow_scores, hard_topK_abn, dim=1)[1]
            hard_topK_feat_abn = torch.gather(hard_topK_feat, 1, hard_topK_idx_abn.unsqueeze(2).expand([-1, -1, F]))
            hard_topK_flow_feat_abn = torch.gather(hard_topK_flow_feat, 1, hard_topK_idx_abn.unsqueeze(2).expand([-1, -1, F]))

            hard_topK_nor = 1    # 从x个困难样本中挑选出x个flow score为正常的样本
            hard_topK_idx_nor = torch.topk(hard_topK_flow_scores, hard_topK_nor, dim=1, largest=False)[1]
            hard_topK_feat_nor = torch.gather(hard_topK_feat, 1, hard_topK_idx_nor.unsqueeze(2).expand([-1, -1, F]))
            hard_topK_flow_feat_nor = torch.gather(hard_topK_flow_feat, 1, hard_topK_idx_nor.unsqueeze(2).expand([-1, -1, F]))


            nor_topK = 2   #将abnormal video中置信度高的x个clip视为normal
            abn_k_idx_nor = torch.topk(abn_scores_rgb, nor_topK, dim=1, largest=False)[1]
            abn_k_idx_nor = abn_k_idx_nor.unsqueeze(2).expand([-1, -1, F])
            abn_rgb_feat_nor = torch.gather(abnormal_rgb, 1, abn_k_idx_nor)
            abn_flow_feat_nor = torch.gather(abnormal_flow, 1, abn_k_idx_nor)

            abn_topK = 10   #挑选出abnormal video中置信度高的x个clip作为abnormal
            abn_k_idx_abn = torch.topk(abn_scores_rgb, abn_topK, dim=1, largest=True)[1]
            abn_k_idx_abn = abn_k_idx_abn.unsqueeze(2).expand([-1, -1, F])
            abn_rgb_feat_abn = torch.gather(abnormal_rgb, 1, abn_k_idx_abn)
            abn_flow_feat_abn = torch.gather(abnormal_flow, 1, abn_k_idx_abn)

            nor_rgb_feat = torch.cat([normal_rgb, hard_topK_feat_nor, abn_rgb_feat_nor], dim=1)  # 在视频内部维度上进行拼接
            nor_flow_feat = torch.cat([normal_flow, hard_topK_flow_feat_nor, abn_flow_feat_nor], dim=1)

            fusion_feat = self.fusion(ref_rgb, ref_flow, nor_rgb_feat, nor_flow_feat)
            ref_attn_feat = fusion_feat
            sup_attn_feat = torch.cat([abn_rgb_feat_abn, hard_topK_feat_abn], dim=1) # todo: rank loss

        else:
            # 当ref为abnormal时
            hard_topK = 10  # 挑选出abnormal ref video中的x个困难样本，即异常分数值接近于0.5
            hard_topK_idx = torch.topk(torch.abs_(ref_p_scores_rgb - 0.5), hard_topK, dim=1, largest=False)[1]
            hard_topK_flow_scores = torch.gather(ref_p_scores_flow, 1, hard_topK_idx)
            hard_topK_feat = torch.gather(ref_rgb, 1, hard_topK_idx.unsqueeze(2).expand([-1, -1, F]))
            hard_topK_flow_feat = torch.gather(ref_flow, 1, hard_topK_idx.unsqueeze(2).expand([-1, -1, F]))

            hard_topK_abn = 1  # 从x个困难样本中挑选出x个flow score为异常的样本
            hard_topK_idx_abn = torch.topk(hard_topK_flow_scores, hard_topK_abn, dim=1)[1]
            hard_topK_feat_abn = torch.gather(hard_topK_feat, 1, hard_topK_idx_abn.unsqueeze(2).expand([-1, -1, F]))
            hard_topK_flow_feat_abn = torch.gather(hard_topK_flow_feat, 1, hard_topK_idx_abn.unsqueeze(2).expand([-1, -1, F]))

            hard_topK_nor = 1  # 从x个困难样本中挑选出x个flow score为正常的样本
            hard_topK_idx_nor = torch.topk(hard_topK_flow_scores, hard_topK_nor, dim=1, largest=False)[1]
            hard_topK_feat_nor = torch.gather(hard_topK_feat, 1, hard_topK_idx_nor.unsqueeze(2).expand([-1, -1, F]))
            hard_topK_flow_feat_nor = torch.gather(hard_topK_flow_feat, 1, hard_topK_idx_nor.unsqueeze(2).expand([-1, -1, F]))

            ##########################
            ref_topK = 10  # 从abnormal ref video中挑选出x个置信度高的异常样本
            ref_k_idx_abn = torch.topk(ref_p_scores_rgb, ref_topK, dim=1, largest=True)[1]
            ref_k_idx_abn = ref_k_idx_abn.unsqueeze(2).expand([-1, -1, F])
            ref_rgb_feat_abn = torch.gather(ref_rgb, 1, ref_k_idx_abn)
            ref_flow_feat_abn = torch.gather(ref_flow, 1, ref_k_idx_abn)

            abn_topK = 10 # 从abnormal sup video中挑选出x个置信度高的异常样本
            abn_k_idx_abn = torch.topk(abn_scores_rgb, abn_topK, dim=1, largest=True)[1]
            abn_k_idx_abn = abn_k_idx_abn.unsqueeze(2).expand([-1, -1, F])
            abn_rgb_feat_abn = torch.gather(abnormal_rgb, 1, abn_k_idx_abn)
            abn_flow_feat_abn = torch.gather(abnormal_flow, 1, abn_k_idx_abn)

            ref_feat_abn = torch.cat([ref_rgb_feat_abn, hard_topK_feat_abn], dim=1)
            ref_flow_abn = torch.cat([ref_flow_feat_abn, hard_topK_flow_feat_abn], dim=1)

            fusion_feat_nor = self.fusion(ref_feat_abn, ref_flow_abn, abn_rgb_feat_abn, abn_flow_feat_abn)
            ref_attn_feat = fusion_feat_nor
            sup_attn_feat = torch.cat([normal_rgb, hard_topK_feat_nor], dim=1) # todo: rank loss

        # ref_attn_feat = self.Aggregate(ref_attn_feat)   #放在这里会起负作用
        # ref_attn_feat = self.drop_out(ref_attn_feat)
        # ref_attn_feat = self.Aggregate(ref_attn_feat[0], ref_attn_feat[1])

        ref_scores = self.f_classifier(ref_attn_feat[0])

        if tencrop:
            ref_p_scores_rgb = ref_p_scores_rgb.view(bs, ncrops, -1).mean(1)
            ref_scores = ref_scores.view(-1, ncrops).mean(1)

        return ref_p_scores_rgb, ref_scores, ref_attn_feat, sup_attn_feat

